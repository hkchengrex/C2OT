import logging
import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from c2ot.ot import OTConditionalPlanSampler
from c2ot.data.memmap_images import MemmapImages
from c2ot.utils.dist_utils import local_rank

log = logging.getLogger()


# Re-seed randomness every time we start a worker
def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    log.debug(f'Worker {worker_id} re-seeded with seed {worker_seed} in rank {local_rank}')


def setup_training_datasets(cfg: DictConfig) -> tuple[Dataset, DistributedSampler, DataLoader]:
    if cfg.dataset == 'cifar':
        # For CIFAR-10, use PyTorch's built-in
        dataset = datasets.CIFAR10(
            cfg.data_path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
    elif cfg.dataset.startswith('imagenet'):
        # For ImageNet, use memmapped images for efficiency
        dataset = MemmapImages(
            cfg.data_path,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )

    batch_size = cfg.batch_size * cfg.oversample_ratio
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory

    # We run OT in the dataloader, so we need to construct the OT planner here
    fm_type = cfg.fm_type
    if fm_type == 'fm':
        ot_planner = None
    elif fm_type == 'ot':
        ot_planner = OTConditionalPlanSampler()
    elif fm_type == 'c2ot':
        ot_planner = OTConditionalPlanSampler(condition_weight=cfg.cls_weight,
                                              target_r=cfg.target_r,
                                              max_num_iter=cfg.max_num_iter,
                                              target_r_tol=cfg.target_r_tol,
                                              x_norm=cfg.ot_norm,
                                              condition_norm=cfg.condition_norm)
    else:
        raise NotImplementedError(f'Unknown fm_type: {fm_type}')

    sampler, loader = construct_loader(dataset,
                                       batch_size,
                                       num_workers,
                                       shuffle=True,
                                       drop_last=True,
                                       pin_memory=pin_memory,
                                       ot_planner=ot_planner)

    return dataset, sampler, loader


def error_avoidance_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


# https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/loader.py
# So the workers are always working between epochs
class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def ot_collate(batch, ot_planner: OTConditionalPlanSampler = None):
    batch = default_collate(batch)

    x1, c = batch
    x0 = torch.randn_like(x1)
    if ot_planner is not None:
        x0, x1, _, c = ot_planner.sample_plan_with_labels(x0, x1, c, c)
    output = {
        'x0': x0.contiguous(),
        'x1': x1.contiguous(),
        'c': c.contiguous(),
    }
    if ot_planner is not None and ot_planner.target_r is not None:
        output['w'] = ot_planner.condition_weight
    return output


def construct_loader(
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
        pin_memory: bool = False,
        ot_planner: OTConditionalPlanSampler = None) -> tuple[DistributedSampler, DataLoader]:
    train_sampler = DistributedSampler(dataset, rank=local_rank, shuffle=shuffle)
    train_loader = MultiEpochsDataLoader(dataset,
                                         batch_size,
                                         sampler=train_sampler,
                                         num_workers=num_workers,
                                         worker_init_fn=worker_init_fn,
                                         drop_last=drop_last,
                                         persistent_workers=num_workers > 0,
                                         pin_memory=pin_memory,
                                         collate_fn=lambda x: ot_collate(x, ot_planner=ot_planner))
    return train_sampler, train_loader
