import os
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import ot
import torch
from torch import optim

from c2ot.ot import OTConditionalPlanSampler
from c2ot.utils.playground_utils import *

total_iter = 20000
batch_size = 256
ot_batch_size = 1024
vis_batch_size = 1024
dim = 128
vis_num_points = 1000

wasserstein_batch_size = 10_000


def train(args, seed: int = -1):
    fm_type = args.type
    condition_type = args.condition
    target_r = args.r
    vis = args.vis
    metric = args.metric

    is_conditional = (condition_type != 'unc')
    if condition_type == 'x' or condition_type == 'y':
        condition_norm = 'l2_squared'
    else:
        condition_norm = 'cosine'

    output_name = f'moons-{fm_type}-{condition_type}'

    if fm_type == 'c2ot':
        if condition_type == 'cls':
            savedir = Path(f'./output/moons-{fm_type}-{condition_type}')
            conditional_weight = 1e6  # effectively infinity
            target_r = None
        else:
            savedir = Path(f'./output/moons-{fm_type}-{condition_type}-{target_r}')
            conditional_weight = 100  # initialization
    else:
        savedir = Path(f'./output/moons-{fm_type}-{condition_type}')
        target_r = None
        conditional_weight = None

    if seed >= 0:
        savedir = savedir / f'seed{seed}'
        torch.manual_seed(seed)
        np.random.seed(seed)

    os.makedirs(savedir, exist_ok=True)

    if is_conditional:
        model = ConditionalNet(in_dim=2, dim=dim, depth=3).cuda()
    else:
        model = SimpleNet(in_dim=2, dim=dim, depth=3).cuda()

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    ot_planner = OTConditionalPlanSampler(
        conditional_weight,
        target_r=target_r,
        condition_norm=condition_norm,
    ) if fm_type != 'fm' else None

    dataset = EightGaussianToMoons(ot_batch_size, condition_type, ot_planner)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=None,
                                             shuffle=False,
                                             num_workers=16)

    # print the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params/1e3:.2f}K')

    start = time.time()
    k = 0
    for x0, x1, y1 in dataloader:

        x0 = x0.cuda()
        x1 = x1.cuda()
        y1 = y1.cuda() if y1 is not None else None

        x0_full_batch = x0
        x1_full_batch = x1
        y1_full_batch = y1

        for i in range(0, ot_batch_size, batch_size):
            optimizer.zero_grad(set_to_none=True)
            x0 = x0_full_batch[i:i + batch_size]
            x1 = x1_full_batch[i:i + batch_size]
            y1 = y1_full_batch[i:i + batch_size] if y1_full_batch is not None else None

            t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)

            xt = sample_conditional_pt(x0, x1, t)
            ut = compute_conditional_vector_field(x0, x1)

            if is_conditional:
                vt = model(xt, y1, t[:, None])
            else:
                vt = model(xt, t[:, None])
            loss = F.mse_loss(vt, ut)

            loss.backward()
            optimizer.step()

            k += 1

            if k % 5000 == 0 or vis:
                end = time.time()
                print(f'{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}')

                # visualize coupling
                val_dataset = EightGaussianToMoons(vis_batch_size, condition_type, ot_planner)
                val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                             batch_size=None,
                                                             shuffle=False,
                                                             num_workers=0)
                x0, x1, y1 = next(iter(val_dataloader))
                traj = np.stack([x0.cpu().numpy(), x1.cpu().numpy()], axis=0)
                if is_conditional:
                    save_trajectories_colored(traj, y1, savedir / f'{output_name}_train_{k}.png')
                else:
                    save_trajectories(traj, savedir / f'{output_name}_train_{k}.png')

                # visualize trajectories
                x0 = sample_8gaussians(vis_batch_size).cuda()
                x1, cls_labels = sample_moons(vis_batch_size)
                x1 = x1.cuda()
                if condition_type == 'cls':
                    y1 = cls_labels.cuda().unsqueeze(-1).float() * 2 - 1
                elif condition_type == 'x':
                    y1 = x1[:, 0:1]
                elif condition_type == 'y':
                    y1 = x1[:, 1:2]
                else:
                    y1 = None
                traj, _ = integrate(model,
                                    x0,
                                    'dopri5',
                                    vis_num_points,
                                    condition=y1,
                                    num_points=vis_num_points)
                if is_conditional:
                    save_trajectories_colored(traj, y1, savedir / f'{output_name}_traj_{k}.jpg')
                else:
                    save_trajectories(traj, savedir / f'{output_name}_traj_{k}.jpg')

                traj, _ = integrate(model, x0, 'euler', 1, condition=y1, num_points=vis_num_points)
                if is_conditional:
                    save_trajectories_colored(traj, y1,
                                              savedir / f'{output_name}_traj_euler1_{k}.jpg')
                else:
                    save_trajectories(traj, savedir / f'{output_name}_traj_euler1_{k}.jpg')

                start = end

            if k >= total_iter:
                break
        if k >= total_iter:
            break

    if metric:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # compute the Wasserstein distance
        x0 = sample_8gaussians(wasserstein_batch_size).cuda()
        x1, cls_labels = sample_moons(wasserstein_batch_size)
        x1 = x1.cuda()
        if condition_type == 'cls':
            y1 = cls_labels.cuda().unsqueeze(-1).float() * 2 - 1
        elif condition_type == 'x':
            y1 = x1[:, 0:1]
        elif condition_type == 'y':
            y1 = x1[:, 1:2]
        else:
            y1 = None

        # adaptive
        all_target_points = []
        all_nfes = []
        for i in range(0, wasserstein_batch_size, batch_size):
            x0_batch = x0[i:i + batch_size]
            y1_batch = y1[i:i + batch_size] if y1 is not None else None
            traj, nfe = integrate(model, x0_batch, 'dopri5', 1, condition=y1_batch)
            target_points = torch.from_numpy(traj[-1]).float().cuda()
            all_target_points.append(target_points)
            all_nfes.append(nfe)
        target_points = torch.cat(all_target_points)
        cost = ot.dist(x1, target_points)
        nfe = np.mean(all_nfes)
        ada_distance = ot.emd2([], [], cost, numItermax=1e8).item()
        print(f'Adaptive wasserstein distance: {ada_distance:.4f}')
        print(f'Number of function evaluations: {nfe}')

        # euler-1
        traj, _ = integrate(model, x0, 'euler', 1, condition=y1)
        target_points = torch.from_numpy(traj[-1]).float().cuda()
        cost = ot.dist(x1, target_points)
        euler1_distance = ot.emd2([], [], cost, numItermax=1e8).item()
        print(f'Euler-1 wasserstein distance: {euler1_distance:.4f}')

        return savedir, ada_distance, euler1_distance, nfe


def main():

    parser = ArgumentParser()
    parser.add_argument('type', type=str, default='fm', choices=['fm', 'ot', 'c2ot'])
    parser.add_argument('condition', type=str, default='unc', choices=['unc', 'cls', 'x', 'y'])
    parser.add_argument('--r', type=float, default=0.05)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--metric', action='store_true')
    args = parser.parse_args()

    metric = args.metric

    if metric:
        ada_distances = []
        euler1_distances = []
        nfes = []
        for seed in range(10):
            savedir, ada_distance, euler1_distance, nfe = train(args, seed)
            ada_distances.append(ada_distance)
            euler1_distances.append(euler1_distance)
            nfes.append(nfe)

        ada_distances = np.array(ada_distances)
        euler1_distances = np.array(euler1_distances)
        nfes = np.array(nfes)
        print(
            f'Adaptive wasserstein distance: {ada_distances.mean():.4f} +- {ada_distances.std():.4f}'
        )
        print(
            f'Euler-1 wasserstein distance: {euler1_distances.mean():.4f} +- {euler1_distances.std():.4f}'
        )
        print(f'Number of function evaluations: {nfes.mean()} +- {nfes.std()}')

        # also save the results to a file
        with open(savedir.parent / 'wasserstein.txt', 'w') as f:
            f.write(f'{ada_distances.mean():.4f}\t{ada_distances.std():.4f}\n')
            f.write(f'{euler1_distances.mean():.4f}\t{euler1_distances.std():.4f}\n')
            f.write(f'{nfes.mean():.4f}\t{nfes.std():.4f}\n')

    else:
        train(args)


if __name__ == '__main__':
    main()
