import logging
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

from c2ot.runner import Runner
from c2ot.utils.dist_utils import info_if_rank_zero, local_rank, world_size
from c2ot.utils.logger import TensorboardLogger

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False

log = logging.getLogger()


def distributed_setup():
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=6))
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size


@record
@hydra.main(version_base='1.3.2', config_path='config', config_name='cifar_train.yaml')
def eval(cfg: DictConfig):
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

    # Set seeds to ensure the same initialization
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # setting up configurations
    info_if_rank_zero(log, f'Configuration: {cfg}')

    conditional = cfg.conditional
    if conditional and cfg.dataset.startswith('imagenet'):
        val_clip_features = torch.load(cfg.val_clip_path, weights_only=True)
    else:
        val_clip_features = None

    # construct the trainer
    trainer = Runner(cfg,
                     log=log,
                     run_path=run_dir,
                     for_training=False,
                     val_clip_feautres=val_clip_features).enter_val()

    # load the checkpoint
    assert cfg.checkpoint is not None
    trainer.load_checkpoint(cfg['checkpoint'])
    cfg['checkpoint'] = None
    info_if_rank_zero(log, 'Model checkpoint loaded!')

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
    eval()
