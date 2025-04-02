import io
import logging
import shutil
from pathlib import Path
from zipfile import ZipFile

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
             num_classes: int,
             num_seeds: int,
             num_gen: int,
             conditional: bool,
             method: str,
             num_steps: int,
             batch_size: int,
             output_path: Path,
             tol: float,
             size: int = 32,
             dataset: str = 'cifar10') -> float:
    network.eval()
    device = next(network.parameters()).device

    num_gen_per_class = num_gen // num_classes
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f'Generating {num_gen} samples; method={method}, num_steps={num_steps}')

    fid_scores = {}
    nfe_total = 0
    nfe_count = 0
    for seed in range(num_seeds):
        zip_dir = output_path / f'seed_{seed}.zip'
        with ZipFile(zip_dir, 'w') as zf:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
            all_classes = torch.arange(num_classes, device=device).repeat(num_gen_per_class)
            if len(all_classes) < num_gen:
                all_classes = torch.cat([
                    all_classes,
                    torch.arange(num_classes, device=device)[:num_gen - len(all_classes)]
                ])
            all_classes_np = all_classes.cpu().numpy()
            for i in tqdm(range(0, num_gen, batch_size)):
                num_samples = min(batch_size, num_gen - i)
                x = torch.randn(num_samples, 3, size, size, device=device, generator=rng)
                if conditional:
                    y = all_classes[i:i + num_samples]
                else:
                    y = None

                if method == 'euler':
                    x = euler_integrate(network, x=x, num_steps=num_steps, y=y)
                    nfe_total += num_steps * num_samples
                    nfe_count += num_samples
                elif method == 'dopri5':
                    t_span = torch.linspace(0, 1, 2, device=device)
                    # network_fn = lambda t, x: network(t, x, y=y)
                    network.nfe = 0

                    def network_fn(t, x):
                        network.nfe += 1
                        return network(t.unsqueeze(0), x, y)

                    x = odeint(network_fn, x, t_span, rtol=tol, atol=tol, method=method)
                    nfe_total += network.nfe * num_samples
                    nfe_count += num_samples
                    x = x[-1, :]

                # 1024, 3, size, size]
                x = (x * 127.5 + 128).clip(0, 255).to(torch.uint8)
                x = x.permute(0, 2, 3, 1).contiguous().view(-1, size, size, 3)
                x = x.cpu().numpy()
                for j in range(num_samples):
                    img = Image.fromarray(x[j])
                    img_byte_array = io.BytesIO()
                    img.save(img_byte_array, format='PNG')
                    ci = all_classes_np[i + j]
                    zf.writestr(f'{ci:04d}_{i + j:09d}.png', img_byte_array.getvalue())

        if dataset == 'cifar':
            score = fid.compute_fid(str(zip_dir),
                                    dataset_name='cifar10',
                                    dataset_res=32,
                                    num_gen=num_gen,
                                    dataset_split='train',
                                    mode='legacy_tensorflow',
                                    use_dataparallel=False)  # crucial for speed
        elif dataset == 'imagenet64':
            score = fid.compute_fid(str(zip_dir),
                                    dataset_name='imagenet64',
                                    dataset_res=64,
                                    num_gen=num_gen,
                                    dataset_split='custom',
                                    use_dataparallel=False)
        elif dataset == 'imagenet32':
            score = fid.compute_fid(str(zip_dir),
                                    dataset_name='imagenet32',
                                    dataset_res=32,
                                    num_gen=num_gen,
                                    dataset_split='custom',
                                    use_dataparallel=False)
        fid_scores[seed] = score
        logging.info('FID', score)

    dir_name = output_path.name
    scores = np.array(list(fid_scores.values()))
    nfes = nfe_total / nfe_count
    logging.info(f'FID: {scores.mean()} ± {scores.std()}')
    logging.info(f'NFE: {nfes}')

    with open(output_path / f'{dir_name}.txt', 'w') as f:
        for seed, score in fid_scores.items():
            f.write(f'{seed} {score}\n')
        # write average and std
        scores = np.array(list(fid_scores.values()))
        f.write(f'average {scores.mean()}\n')
        f.write(f'std {scores.std()}\n')
        f.write(f'NFE {nfes}\n')

    return scores.mean()


@torch.inference_mode()
def evaluate_ddp(network: nn.Module,
                 *,
                 num_classes: int = None,
                 num_seeds: int,
                 num_gen: int,
                 conditional: bool,
                 method: str,
                 num_steps: int,
                 batch_size: int,
                 output_path: Path,
                 tol: float,
                 size: int = 32,
                 dataset: str = 'cifar10',
                 class_features: dict[str, torch.Tensor] = None) -> float:
    network.eval()
    device = next(network.parameters()).device

    if class_features is None:
        num_gen_per_class = num_gen // num_classes
        all_classes = torch.arange(num_classes, device=device).repeat(num_gen_per_class)
        if len(all_classes) < num_gen:
            all_classes = torch.cat([
                all_classes,
                torch.arange(num_classes, device=device)[:num_gen - len(all_classes)]
            ])
        all_classes_np = all_classes.cpu().numpy()
    else:
        all_file_ids = list(class_features.keys())

    # assign the classes to each device
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
            num_samples = min(batch_size, num_gen - start_gen_idx)
            x = torch.randn(num_samples, 3, size, size, device=device, generator=rng)

        for i in tqdm(range(start_gen_idx, end_gen_idx, batch_size)):
            num_samples = min(batch_size, num_gen - i)
            x = torch.randn(num_samples, 3, size, size, device=device, generator=rng)

            if class_features is not None:
                file_ids = all_file_ids[i:i + num_samples]
                y = torch.stack([class_features[file_id] for file_id in file_ids]).to(device)
            else:
                if conditional:
                    y = all_classes[i:i + num_samples]
                else:
                    y = None

            if method == 'euler':
                x = euler_integrate(network, x=x, num_steps=num_steps, y=y)
                nfe_total += num_steps * num_samples
                nfe_count += num_samples
            elif method == 'dopri5':
                t_span = torch.linspace(0, 1, 2, device=device)
                # network_fn = lambda t, x: network(t, x, y=y)
                network.nfe = 0

                def network_fn(t, x):
                    network.nfe += 1
                    return network(t.unsqueeze(0), x, y)

                x = odeint(network_fn, x, t_span, rtol=tol, atol=tol, method=method)
                nfe_total += network.nfe * num_samples
                nfe_count += num_samples
                x = x[-1, :]

            # 1024, 3, size, size]
            x = (x * 127.5 + 128).clip(0, 255).to(torch.uint8)
            x = x.permute(0, 2, 3, 1).contiguous().view(-1, size, size, 3)
            x = x.cpu().numpy()
            for j in range(num_samples):
                # write as image
                img = Image.fromarray(x[j])
                if class_features is not None:
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
                elif dataset == 'imagenet64':
                    score = fid.compute_fid(str(seed_zip_dir),
                                            dataset_name='imagenet64_val',
                                            dataset_res=64,
                                            num_gen=num_gen,
                                            dataset_split='custom',
                                            mode='legacy_tensorflow',
                                            use_dataparallel=False)
                elif dataset == 'imagenet32':
                    score = fid.compute_fid(str(seed_zip_dir),
                                            dataset_name='imagenet32_val',
                                            dataset_res=32,
                                            num_gen=num_gen,
                                            dataset_split='custom',
                                            mode='legacy_tensorflow',
                                            use_dataparallel=False)
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
