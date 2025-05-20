import logging
import math
import random
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as distributed
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.distributed.elastic.multiprocessing.errors import record

from c2ot.data.data_setup import setup_training_datasets
from c2ot.runner import Runner
from c2ot.utils.dist_utils import info_if_rank_zero, local_rank, world_size
from c2ot.utils.logger import TensorboardLogger

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

log = logging.getLogger()


def distributed_setup():
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=6))
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size


@record
@hydra.main(version_base='1.3.2', config_path='config', config_name='cifar_train.yaml')
def train(cfg: DictConfig):
    # initial setup
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    distributed_setup()
    num_gpus = world_size
    run_dir = HydraConfig.get().run.dir

    # wrap python logger with a tensorboard logger
    log = TensorboardLogger(cfg.exp_id, run_dir, logging.getLogger(), is_rank0=(local_rank == 0))

    info_if_rank_zero(log, f'All configuration: {cfg}')
    info_if_rank_zero(log, f'Number of GPUs detected: {num_gpus}')

    # number of dataloader workers
    info_if_rank_zero(log, f'Number of dataloader workers (per GPU): {cfg.num_workers}')

    # Set seeds to ensure the same initialization
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # setting up configurations
    info_if_rank_zero(log, f'Training configuration: {cfg}')
    cfg.batch_size //= num_gpus
    ot_batch_size = cfg.batch_size * cfg.oversample_ratio
    info_if_rank_zero(log, f'Batch size (per GPU): {cfg.batch_size}')
    info_if_rank_zero(log, f'OT batch size (per GPU): {ot_batch_size}')

    # determine time to change max skip
    total_iterations = cfg['num_iterations']

    # setup datasets
    dataset, sampler, loader = setup_training_datasets(cfg)
    info_if_rank_zero(log, f'Number of training samples: {len(dataset)}')
    info_if_rank_zero(log, f'Number of training batches: {len(loader)}')

    conditional = cfg.conditional
    if conditional and cfg.dataset.startswith('imagenet'):
        val_clip_features = torch.load(cfg.val_clip_path, weights_only=True)
    else:
        val_clip_features = None

    # construct the trainer
    trainer = Runner(cfg,
                     log=log,
                     run_path=run_dir,
                     for_training=True,
                     val_clip_feautres=val_clip_features).enter_train()
    eval_rng_clone = trainer.rng.graphsafe_get_state()

    # load previous checkpoint if needed
    if cfg['checkpoint'] is not None:
        curr_iter = trainer.load_checkpoint(cfg['checkpoint'])
        cfg['checkpoint'] = None
        info_if_rank_zero(log, 'Model checkpoint loaded!')
    else:
        # if run_dir exists, load the latest checkpoint
        checkpoint = trainer.get_latest_checkpoint_path()
        if checkpoint is not None:
            curr_iter = trainer.load_checkpoint(checkpoint)
            info_if_rank_zero(log, 'Latest checkpoint loaded!')
        else:
            curr_iter = 0

    batch_size = cfg.batch_size
    # determine max epoch
    total_epoch = math.ceil(total_iterations / len(loader) / (ot_batch_size // batch_size))
    current_epoch = curr_iter // len(loader) // (ot_batch_size // batch_size)
    info_if_rank_zero(log, f'We will approximately use {total_epoch} epochs.')

    # training loop
    try:
        # Need this to select random bases in different workers
        np.random.seed(np.random.randint(2**30 - 1) + local_rank * 1000)
        while curr_iter < total_iterations:
            # Crucial for randomness!
            sampler.set_epoch(current_epoch)
            current_epoch += 1
            info_if_rank_zero(f'Current epoch: {current_epoch}')
            rng = torch.Generator()
            rng.manual_seed(cfg['seed'] + local_rank)

            trainer.enter_train()
            trainer.log.data_timer.start()
            for data in loader:
                # a oversampled batch
                x0, x1, c = data['x0'], data['x1'], data['c']
                if 'w' in data:
                    other_things_to_log = {
                        'w': data['w'],
                    }
                else:
                    other_things_to_log = None
                if not conditional:
                    c = None

                for bs_start in range(0, len(x0), batch_size):
                    # for each minibatch in the oversampled batch
                    if bs_start + batch_size > len(x0):
                        break
                    x0_batch = x0[bs_start:bs_start + batch_size]
                    x1_batch = x1[bs_start:bs_start + batch_size]
                    if c is not None:
                        c_batch = c[bs_start:bs_start + batch_size]
                    else:
                        c_batch = None
                    trainer.train_pass(x0_batch,
                                       x1_batch,
                                       c_batch,
                                       curr_iter,
                                       other_things_to_log=other_things_to_log)

                    if (curr_iter + 1) % cfg.val_interval == 0:
                        # get validation FID to track training progress
                        train_rng_snapshot = trainer.rng.graphsafe_get_state()
                        trainer.rng.graphsafe_set_state(eval_rng_clone)
                        info_if_rank_zero(log, f'Iteration {curr_iter}: validating')
                        trainer.validation_pass(curr_iter)
                        distributed.barrier()
                        trainer.rng.graphsafe_set_state(train_rng_snapshot)

                    curr_iter += 1

                    if curr_iter >= total_iterations:
                        break

                if curr_iter >= total_iterations:
                    break
    except Exception as e:
        log.error(f'Error occurred at iteration {curr_iter}!')
        log.critical(e.message if hasattr(e, 'message') else str(e))
        raise
    finally:
        if not cfg.debug:
            trainer.save_checkpoint(curr_iter)
            trainer.save_weights(curr_iter)

    distributed.barrier()

    # final eval
    ema_model = trainer.ema
    methods = ['euler_2', 'euler_5', 'euler_10', 'euler_25', 'euler_50', 'euler_100', 'dopri5']

    for method in methods:
        log.info(f'Running evaluation for {method}')
        fid = trainer.evaluate(
            network=ema_model,
            num_seeds=1,
            method='euler' if method.startswith('euler') else method,
            num_steps=int(method.split('_')[-1]) if method.startswith('euler') else 0,
            output_path=Path(run_dir) / 'final' / method)
        if fid is not None:
            log.info(f'FID for {method}: {fid}')

    # clean-up
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == '__main__':
    train()
