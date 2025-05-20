import logging
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as distributed
import torch.nn as nn
from cleanfid import fid
from PIL import Image
from torchdiffeq import odeint
from tqdm import tqdm

from c2ot.utils.dist_utils import info_if_rank_zero, local_rank, world_size
from c2ot.utils.gen_utils import euler_integrate

log = logging.getLogger()


@torch.inference_mode()
def evaluate(network: nn.Module,
             *,
             num_classes: int = None,
             num_seeds: int,
             num_gen: int,
             conditional: bool,
             method: Literal['euler', 'dopri5'],
             num_steps: int,
             tol: float,
             batch_size: int,
             output_path: Path,
             size: int = 32,
             dataset: Literal['cifar10', 'imagenet32'] = 'cifar10',
             clip_features: dict[str, torch.Tensor] = None) -> float:
    """
    Evaluate the model using FID score (distributed).
    Args:
        network (nn.Module): The model. Its forward function should take (t, x, (optional) y)
        num_classes (int): The number of classes for class-conditional generation.
        num_seeds (int): The number of seeds to use. We repeat generation for each seed.
        num_gen (int): The number of samples to generate, per seed.
        conditional (bool): Whether the model is conditional or not.
        method (str): 'euler' or 'dopri5'.
        num_steps (int): The number of steps for the euler method.
        tol (float): The tolerance for the ODE solver for the 'dopri5' method.
        batch_size (int): The batch size to use for generation.
        output_path (Path): The path to save the generated images (as zip files).
        size (int): The size of the generated images.
        dataset (str): The dataset to use. Can be 'cifar10' or 'imagenet32'.
        clip_features (dict[str, torch.Tensor]): A dictionary mapping file ids to condition features.
    """
    network.eval()
    device = next(network.parameters()).device

    if conditional:
        if clip_features is None:
            # class-conditional generation
            # Figuring out the class ids to use for generation.
            num_gen_per_class = num_gen // num_classes
            all_classes = torch.arange(num_classes, device=device).repeat(num_gen_per_class)
            if len(all_classes) < num_gen:
                all_classes = torch.cat([
                    all_classes,
                    torch.arange(num_classes, device=device)[:num_gen - len(all_classes)]
                ])
            all_classes_np = all_classes.cpu().numpy()
        else:
            # CLIP-conditional generation
            all_file_ids = list(clip_features.keys())

    # assign the generations to each device
    num_gen_per_device = num_gen // world_size
    start_gen_idx = local_rank * num_gen_per_device
    end_gen_idx = start_gen_idx + num_gen_per_device
    if local_rank == (world_size - 1):
        # let the last device handle the remaining samples
        end_gen_idx = num_gen

    logging.info(
        f'Generating from {start_gen_idx} to {end_gen_idx}; method={method}, num_steps={num_steps}')
    distributed.barrier()

    fid_scores = {}
    nfe_total = 0
    nfe_count = 0
    for seed in range(num_seeds):
        seed_output_path = output_path / f'seed_{seed}'
        seed_output_path.mkdir(parents=True, exist_ok=True)
        seed_zip_dir = output_path / f'seed_{seed}'

        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        for _ in range(0, start_gen_idx, batch_size):
            # consume the rng states to match the start_gen_idx
            # otherwise all devices will generate with the same rng state
            num_samples = min(batch_size, num_gen - start_gen_idx)
            x = torch.randn(num_samples, 3, size, size, device=device, generator=rng)

        for i in tqdm(range(start_gen_idx, end_gen_idx, batch_size)):
            num_samples = min(batch_size, num_gen - i)
            x = torch.randn(num_samples, 3, size, size, device=device, generator=rng)

            if conditional:
                if clip_features is None:
                    # class-conditional
                    y = all_classes[i:i + num_samples]
                else:
                    # CLIP-conditional
                    file_ids = all_file_ids[i:i + num_samples]
                    y = torch.stack([clip_features[file_id] for file_id in file_ids]).to(device)
            else:
                # unconditional
                y = None

            if method == 'euler':
                x = euler_integrate(network, x=x, num_steps=num_steps, y=y)
                nfe_total += num_steps * num_samples
                nfe_count += num_samples
            elif method == 'dopri5':
                t_span = torch.linspace(0, 1, 2, device=device)
                network.nfe = 0

                def network_fn(t, x):
                    network.nfe += 1
                    return network(t.unsqueeze(0), x, y)

                x = odeint(network_fn, x, t_span, rtol=tol, atol=tol, method=method)
                nfe_total += network.nfe * num_samples
                nfe_count += num_samples
                x = x[-1, :]

            # unnormalize and saev the images
            x = (x * 127.5 + 128).clip(0, 255).to(torch.uint8)
            x = x.permute(0, 2, 3, 1).contiguous().view(-1, size, size, 3)
            x = x.cpu().numpy()
            for j in range(num_samples):
                # write as image
                img = Image.fromarray(x[j])
                if clip_features is not None:
                    img_path = seed_output_path / f'{file_ids[j]}.png'
                else:
                    ci = all_classes_np[i + j]
                    img_path = seed_output_path / f'{ci:04d}_{i + j:09d}.png'
                img.save(img_path)

        # wait for all devices to finish generation
        distributed.barrier()

        with torch.amp.autocast('cuda', enabled=False):
            if local_rank == 0:
                number_of_files = len(list(seed_output_path.iterdir()))
                info_if_rank_zero(log, f'Number of files: {number_of_files}')
                # zip the images
                shutil.make_archive(seed_zip_dir, 'zip', seed_output_path)

                if dataset == 'cifar':
                    score = fid.compute_fid(str(seed_zip_dir),
                                            dataset_name='cifar10',
                                            dataset_res=32,
                                            num_gen=num_gen,
                                            dataset_split='train',
                                            mode='legacy_tensorflow',
                                            use_dataparallel=False)  # crucial for speed
                elif dataset == 'imagenet32':
                    score = fid.compute_fid(str(seed_zip_dir),
                                            dataset_name='imagenet32_val',
                                            dataset_res=32,
                                            num_gen=num_gen,
                                            dataset_split='custom',
                                            mode='legacy_tensorflow',
                                            use_dataparallel=False)  # crucial for speed
                fid_scores[seed] = score
                info_if_rank_zero(log, f'FID: {score}')

                # remove the images
                shutil.rmtree(seed_output_path)

    distributed.barrier()

    if local_rank == 0:
        dir_name = output_path.name
        scores = np.array(list(fid_scores.values()))
        nfes = nfe_total / nfe_count
        info_if_rank_zero(log, f'FID: {scores.mean()} ± {scores.std()}')
        info_if_rank_zero(log, f'NFE: {nfes}')

        with open(output_path / f'{dir_name}.txt', 'w') as f:
            for seed, score in fid_scores.items():
                f.write(f'{seed} {score}\n')
            # write average and std
            scores = np.array(list(fid_scores.values()))
            f.write(f'average {scores.mean()}\n')
            f.write(f'std {scores.std()}\n')
            f.write(f'NFE {nfes}\n')

    if local_rank == 0:
        return scores.mean()
    else:
        return None
