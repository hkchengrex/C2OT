"""
runner.py - wrapper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""
import copy
import os
from functools import partial
from pathlib import Path
from typing import Optional, Union

import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from c2ot.evaluate import evaluate
from c2ot.model.unet import UNetModelWrapper
from c2ot.utils.dist_utils import info_if_rank_zero, local_rank, string_if_rank_zero
from c2ot.utils.gen_utils import ema_update, generate_samples
from c2ot.utils.log_integrator import Integrator
from c2ot.utils.logger import TensorboardLogger
from c2ot.utils.time_estimator import PartialTimeEstimator, TimeEstimator


class Runner:

    def __init__(
        self,
        cfg: DictConfig,
        log: TensorboardLogger,
        run_path: Union[str, Path],
        for_training: bool = True,
        val_clip_feautres: dict[str, torch.Tensor] = None,
    ):
        self.exp_id = cfg.exp_id
        self.use_amp = cfg.amp
        self.for_training = for_training
        self.cfg = cfg
        self.val_clip_features = val_clip_feautres
        self.num_gen = cfg.num_gen
        if self.val_clip_features is not None:
            for k, v in self.val_clip_features.items():
                self.val_clip_features[k] = v.cuda()

        self.continuous_code = (cfg.conditioning_code == 'continuous')
        self.dataset = cfg.dataset
        self.size = cfg.size
        self.num_classes = cfg.num_classes
        self.conditional = cfg.conditional

        self.evaluate = partial(evaluate,
                                num_classes=cfg.num_classes,
                                conditional=self.conditional,
                                size=cfg.size,
                                batch_size=cfg.eval_batch_size,
                                dataset=cfg.dataset,
                                num_gen=self.num_gen,
                                tol=1e-5,
                                clip_features=self.val_clip_features)

        # setting up the model
        if cfg.model == 'unet-cifar':
            network = UNetModelWrapper(
                dim=(3, self.size, self.size),
                num_res_blocks=2,
                num_channels=192,
                channel_mult=[1, 2, 2, 2],
                num_heads=4,
                num_head_channels=64,
                attention_resolutions='16,8',
                dropout=cfg.dropout,
                num_classes=self.num_classes,
                use_scale_shift_norm=True,
                continuous_code=self.continuous_code,
                code_dim=cfg.code_dim,
                conditional=self.conditional,
            )
        elif cfg.model == 'unet-imagenet':
            network = UNetModelWrapper(
                dim=(3, self.size, self.size),
                num_res_blocks=3,
                num_channels=256,
                channel_mult=[1, 2, 2, 2],
                num_heads=4,
                num_head_channels=64,
                attention_resolutions='4',
                dropout=cfg.dropout,
                num_classes=self.num_classes,
                use_scale_shift_norm=True,
                continuous_code=self.continuous_code,
                code_dim=cfg.code_dim,
                conditional=self.conditional,
            )
        else:
            raise NotImplementedError

        self.network = DDP(network.cuda(), device_ids=[local_rank], broadcast_buffers=False)
        self.parameters = list(self.network.parameters())

        if cfg.compile:
            self.train_fn = torch.compile(self.train_fn)

        # ema profile
        self.ema = copy.deepcopy(self.network.module)
        self.ema_decay = cfg.ema_decay
        self.num_classes = cfg.num_classes
        self.size = cfg.size

        self.rng = torch.Generator(device='cuda')
        self.rng.manual_seed(cfg.seed + local_rank)

        # setting up logging
        self.log = log
        self.run_path = Path(run_path)

        string_if_rank_zero(self.log, 'model_size',
                            f'{sum([param.nelement() for param in self.parameters])}')
        string_if_rank_zero(
            self.log, 'number_of_parameters_that_require_gradient: ',
            str(
                sum([
                    param.nelement() for param in filter(lambda p: p.requires_grad, self.parameters)
                ])))
        info_if_rank_zero(self.log, 'torch version: ' + torch.__version__)
        self.train_integrator = Integrator(self.log, distributed=True)
        if local_rank == 0:
            self.val_integrator = Integrator(self.log, distributed=False)

        # setting up optimizer and loss
        if for_training:
            self.enter_train()

            base_lr = cfg.learning_rate
            self.optimizer = optim.Adam(self.parameters,
                                        lr=base_lr,
                                        betas=(cfg.adam_beta1, cfg.adam_beta2),
                                        weight_decay=cfg.weight_decay,
                                        fused=True,
                                        eps=1e-6)
            if self.use_amp:
                self.scaler = torch.amp.GradScaler(init_scale=2048, enabled=cfg.enable_grad_scaler)
            self.clip_grad_norm = cfg.clip_grad_norm

            # linearly warmup learning rate
            linear_warmup_steps = cfg.linear_warmup_steps
            min_lr = cfg.min_lr
            min_lr_factor = min_lr / base_lr

            warmup_scheduler = optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=min_lr_factor,
                                                           end_factor=1,
                                                           total_iters=linear_warmup_steps)

            # setting up learning rate scheduler
            if cfg['lr_schedule'] == 'constant':
                next_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1)
            elif cfg['lr_schedule'] == 'step':
                next_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                cfg.lr_schedule_steps,
                                                                cfg.lr_schedule_gamma)
            elif cfg['lr_schedule'] == 'linear':
                next_scheduler = optim.lr_scheduler.LinearLR(self.optimizer,
                                                             start_factor=1,
                                                             end_factor=min_lr_factor,
                                                             total_iters=cfg.num_iterations -
                                                             linear_warmup_steps)
            else:
                raise NotImplementedError

            self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer,
                                                             [warmup_scheduler, next_scheduler],
                                                             [linear_warmup_steps])

            # Logging info
            self.log_text_interval = cfg.log_text_interval
            self.log_extra_interval = cfg.log_extra_interval
            self.save_checkpoint_interval = cfg.save_checkpoint_interval
            self.save_copy_iterations = cfg.save_copy_iterations
            self.save_copy_always = cfg.save_copy_always
            self.num_iterations = cfg.num_iterations
            if cfg.debug:
                self.log_text_interval = self.log_extra_interval = 1

            # update() is called when we log metrics, within the logger
            self.log.batch_timer = TimeEstimator(self.num_iterations, self.log_text_interval)
            # update() is called every iteration, in this script
            self.log.data_timer = PartialTimeEstimator(self.num_iterations, 1, ema_alpha=0.9)
        else:
            self.enter_val()

    def train_fn(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        bs = x1.shape[0]

        t = torch.rand(bs, device=x1.device, dtype=x1.dtype, generator=self.rng)
        t = t.view(-1, 1, 1, 1)

        with torch.amp.autocast('cuda', enabled=False):
            xt = t * x1 + (1 - t) * x0
            ut = x1 - x0

        pred_v = self.network(t.squeeze(), xt, c)

        with torch.amp.autocast('cuda', enabled=False):
            loss = ((pred_v.float() - ut)**2)
            mean_loss = loss.mean()
        return x1, mean_loss, t

    def train_pass(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        it: int = 0,
        *,
        other_things_to_log: dict[str, float] = None,
    ):

        if not self.for_training:
            raise ValueError('train_pass() should not be called when not training.')

        self.enter_train()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            x0 = x0.cuda(non_blocking=True)
            x1 = x1.cuda(non_blocking=True)
            if c is not None:
                c = c.cuda(non_blocking=True)

            self.log.data_timer.end()
            # forward pass
            x1, mean_loss, t = self.train_fn(x0, x1, c)

            self.train_integrator.add_dict({'loss': mean_loss})

        if it % self.log_text_interval == 0 and it != 0:
            self.train_integrator.add_scalar('lr', self.scheduler.get_last_lr()[0])
            if other_things_to_log is not None:
                for k, v in other_things_to_log.items():
                    self.train_integrator.add_scalar(k, v)
            self.train_integrator.finalize('train', it)
            self.train_integrator.reset_except_hooks()

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(mean_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                       self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            mean_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                       self.clip_grad_norm)
            self.optimizer.step()

        if self.ema is not None:
            ema_update(self.network.module, self.ema, self.ema_decay)
        self.scheduler.step()
        self.integrator.add_scalar('grad_norm', grad_norm)

        self.enter_val()
        with (torch.amp.autocast('cuda', enabled=self.use_amp,
                                 dtype=torch.bfloat16), torch.inference_mode()):
            # generate some samples for visualization and performance tracking
            try:
                if it % self.log_extra_interval == 0:
                    generate_samples(
                        self.network.module,
                        self.run_path / 'samples',
                        it,
                        num_classes=self.num_classes,
                        size=self.size,
                        conditional=self.conditional,
                        class_features=self.val_clip_features,
                    )
            except Exception as e:
                self.log.warning(f'Error in extra logging: {e}')
                if self.cfg.debug:
                    raise

        # Save network checkpoint if needed
        save_copy = (it in self.save_copy_iterations) or self.save_copy_always
        if it % self.save_checkpoint_interval == 0 and it != 0:
            self.save_checkpoint(it, save_copy=save_copy)

        self.log.data_timer.start()

    @torch.inference_mode()
    def validation_pass(self, it: int) -> Path:
        self.enter_val()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            fid_normal = self.evaluate(
                network=self.network.module,
                num_seeds=1,
                method='euler',
                num_steps=10,
                output_path=self.run_path / f'eval_{it}',
            )
            fid_ema = self.evaluate(
                network=self.ema,
                num_seeds=1,
                method='euler',
                num_steps=10,
                output_path=self.run_path / f'eval_ema_{it}',
            )
            if local_rank == 0:
                self.val_integrator.add_scalar('fid_normal', fid_normal)
                self.val_integrator.add_scalar('fid_ema', fid_ema)
                self.val_integrator.finalize('val', it)
                self.val_integrator.reset_except_hooks()

    def get_latest_checkpoint_path(self):
        ckpt_path = self.run_path / f'{self.exp_id}_ckpt_last.pth'
        if not ckpt_path.exists():
            info_if_rank_zero(self.log, f'No checkpoint found at {ckpt_path}.')
            return None
        return ckpt_path

    def save_checkpoint(self, it, save_copy=False):
        if local_rank != 0:
            return

        checkpoint = {
            'it': it,
            'weights': self.network.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema': self.ema.state_dict() if self.ema is not None else None,
        }

        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = self.run_path / f'{self.exp_id}_ckpt_{it}.pth'
            torch.save(checkpoint, model_path)
            self.log.info(f'Checkpoint saved to {model_path}.')

        # if ckpt_last exists, move it to a shadow copy
        model_path = self.run_path / f'{self.exp_id}_ckpt_last.pth'
        if model_path.exists():
            shadow_path = model_path.with_name(model_path.name.replace('last', 'shadow'))
            model_path.replace(shadow_path)  # moves the file
            self.log.info(f'Checkpoint shadowed to {shadow_path}.')

        torch.save(checkpoint, model_path)
        self.log.info(f'Checkpoint saved to {model_path}.')

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location}, weights_only=True)

        it = checkpoint['it']
        weights = checkpoint['weights']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        if self.ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])

        map_location = 'cuda:%d' % local_rank
        self.network.module.load_state_dict(weights)
        if self.for_training:
            self.optimizer.load_state_dict(optimizer)
            self.scheduler.load_state_dict(scheduler)

            self.log.info(f'Global iteration {it} loaded.')
            self.log.info('Network weights, optimizer states, and scheduler states loaded.')
        else:
            self.log.info('Network weights and ema weights loaded.')

        return it

    def enter_train(self):
        self.integrator = self.train_integrator
        self.network.train()
        return self

    def enter_val(self):
        self.network.eval()
        return self
