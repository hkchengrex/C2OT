import math
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from c2ot.ot import OTConditionalPlanSampler

# partially from torchcfm https://github.com/atong01/conditional-flow-matching/tree/main


class EightGaussianToMoons(Dataset):

    def __init__(self,
                 batch_size: int,
                 condition_type: Literal['cls', 'x', 'y'],
                 ot_planner: Optional[OTConditionalPlanSampler] = None):
        self.batch_size = batch_size
        self.condition_type = condition_type
        self.ot_planner = ot_planner

    def __len__(self):
        return 100_000

    def __getitem__(self, idx):
        x0 = sample_8gaussians(self.batch_size)
        x1, cls_labels = sample_moons(self.batch_size)

        if self.condition_type == 'cls':
            y1 = cls_labels.unsqueeze(-1).float() * 2 - 1
        elif self.condition_type == 'x':
            y1 = x1[:, 0:1]
        elif self.condition_type == 'y':
            y1 = x1[:, 1:2]
        else:
            y1 = None

        if self.ot_planner is not None:
            x0, x1, _, y1 = self.ot_planner.sample_plan_with_labels(x0, x1, y1, y1)

        return x0, x1, y1


def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim),
                                                                   math.sqrt(var) * torch.eye(dim))
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n, ))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def sample_moons(n):
    x0, y0 = generate_moons(n, noise=0.2)
    return x0 * 3 - 1, y0


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()


@torch.inference_mode()
def integrate(model,
              x0: torch.Tensor,
              solver: str,
              num_steps: int,
              interpolate: bool = True,
              condition: torch.Tensor = None,
              num_points: int = 100) -> np.ndarray:

    if condition is None:
        # unconditional
        wrapper = torch_wrapper(model)
    else:
        wrapper = conditional_torch_wrapper(model, condition)

    node = NeuralODE(wrapper,
                     solver=solver,
                     sensitivity='adjoint',
                     atol=1e-4 if solver == 'dopri5' else 1e-3,
                     rtol=1e-4 if solver == 'dopri5' else 1e-3)
    traj = node.trajectory(
        x0.cuda(),
        t_span=torch.linspace(0, 1, num_steps + 1).cuda(),
    )
    if interpolate and num_steps < num_points:
        traj = F.interpolate(traj.transpose(0, 2), num_points, mode='linear',
                             align_corners=True).transpose(0, 2)
    traj = traj.cpu().numpy()
    return traj, wrapper.nfe


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.nfe = 0

    def forward(self, t, x, *args, **kwargs):
        # return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
        self.nfe += 1
        return self.model(x, t.repeat(x.shape[0])[:, None])


class conditional_torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, c):
        super().__init__()
        self.model = model
        self.c = c
        self.nfe = 0

    def forward(self, t, x, *args, **kwargs):
        # return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
        self.nfe += 1
        return self.model(x, self.c, t.repeat(x.shape[0])[:, None])


def sample_conditional_pt(x0, x1, t):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)
    """
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    return mu_t


def compute_conditional_vector_field(x0, x1):
    return x1 - x0


def save_trajectories(traj: torch.Tensor, save_path: Path):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)

    plt.plot(traj[:, :n, 0], traj[:, :n, 1], alpha=0.7, c='olive', linewidth=0.5, zorder=0)
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=16, alpha=0.8, c='#F4A582', marker='x', zorder=10)
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=8, alpha=1, c='#B2182B', zorder=10)
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    plt.gca().set_frame_on(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, pad_inches=0, bbox_inches='tight')
    plt.close()


N = 256
vals = np.ones((N * 3, 4))
vals[0:N, 0] = np.linspace(178 / 256, 118 / 256, N)
vals[0:N, 1] = np.linspace(24 / 256, 42 / 256, N)
vals[0:N, 2] = np.linspace(32 / 256, 131 / 256, N)
vals[N:2 * N, 0] = np.linspace(118 / 256, 33 / 256, N)
vals[N:2 * N, 1] = np.linspace(42 / 256, 102 / 256, N)
vals[N:2 * N, 2] = np.linspace(131 / 256, 172 / 256, N)
vals[2 * N:3 * N, 0] = np.linspace(33 / 256, 27 / 255, N)
vals[2 * N:3 * N, 1] = np.linspace(102 / 256, 120 / 255, N)
vals[2 * N:3 * N, 2] = np.linspace(172 / 256, 55 / 255, N)
my_colormap = ListedColormap(vals)


def save_trajectories_colored(traj: torch.Tensor, x: torch.Tensor, save_path: Path):
    """Plot trajectories of some selected samples with color based on x."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)

    x = x.cpu().numpy()

    if np.unique(x.flatten()).shape[0] == 2:
        traj1 = traj[:, x.flatten() == -1]
        traj2 = traj[:, x.flatten() == 1]

        plt.plot(traj1[:, :n, 0], traj1[:, :n, 1], alpha=0.7, c='olive', linewidth=0.5, zorder=0)
        plt.plot(traj2[:, :n, 0], traj2[:, :n, 1], alpha=0.7, c='olive', linewidth=0.5, zorder=0)
        plt.scatter(traj1[0, :n, 0],
                    traj1[0, :n, 1],
                    s=16,
                    alpha=1,
                    c='#F4A582',
                    marker='x',
                    zorder=10)
        plt.scatter(traj2[0, :n, 0],
                    traj2[0, :n, 1],
                    s=16,
                    alpha=1,
                    c='#92C5DE',
                    marker='x',
                    zorder=10)
        plt.scatter(traj1[-1, :n, 0], traj1[-1, :n, 1], s=8, alpha=1, c='#B2182B', zorder=10)
        plt.scatter(traj2[-1, :n, 0], traj2[-1, :n, 1], s=8, alpha=1, c='#2166AC', zorder=10)
    else:
        plt.plot(traj[:, :n, 0], traj[:, :n, 1], alpha=0.7, c='olive', linewidth=0.5, zorder=0)
        plt.scatter(traj[0, :n, 0],
                    traj[0, :n, 1],
                    s=16,
                    alpha=0.56,
                    c=x[:n],
                    marker='x',
                    cmap=my_colormap)
        plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=8, alpha=1, c=x[:n], cmap=my_colormap)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, pad_inches=0, bbox_inches='tight')
    plt.close()


def vis_trajectories_colored(traj: torch.Tensor, x: torch.Tensor):
    """Plot trajectories of some selected samples with color based on x."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)

    x = x.cpu().numpy()

    if np.unique(x.flatten()).shape[0] == 2:
        traj1 = traj[:, x.flatten() == -1]
        traj2 = traj[:, x.flatten() == 1]

        plt.plot(traj1[:, :n, 0], traj1[:, :n, 1], alpha=0.7, c='olive', linewidth=0.5, zorder=0)
        plt.plot(traj2[:, :n, 0], traj2[:, :n, 1], alpha=0.7, c='olive', linewidth=0.5, zorder=0)
        plt.scatter(traj1[0, :n, 0],
                    traj1[0, :n, 1],
                    s=16,
                    alpha=1,
                    c='#F4A582',
                    marker='x',
                    zorder=10)
        plt.scatter(traj2[0, :n, 0],
                    traj2[0, :n, 1],
                    s=16,
                    alpha=1,
                    c='#92C5DE',
                    marker='x',
                    zorder=10)
        plt.scatter(traj1[-1, :n, 0], traj1[-1, :n, 1], s=8, alpha=1, c='#B2182B', zorder=10)
        plt.scatter(traj2[-1, :n, 0], traj2[-1, :n, 1], s=8, alpha=1, c='#2166AC', zorder=10)
    else:
        plt.plot(traj[:, :n, 0], traj[:, :n, 1], alpha=0.7, c='olive', linewidth=0.5, zorder=0)
        plt.scatter(traj[0, :n, 0],
                    traj[0, :n, 1],
                    s=16,
                    alpha=0.56,
                    c=x[:n],
                    marker='x',
                    cmap=my_colormap)
        plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=8, alpha=1, c=x[:n], cmap=my_colormap)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    plt.tight_layout()
    plt.show()


def save_trajectories_colored_1d(traj: torch.Tensor, x: torch.Tensor, save_path: Path):
    """Plot trajectories of some selected samples with color based on x."""
    n = 2000
    plt.figure(figsize=(3, 4))
    plt.xlim(-4, 4)
    plt.ylim(-10, 10)

    num_points = traj.shape[0]
    y_axis = np.linspace(-9, 9, num_points).reshape(num_points, 1).repeat(traj.shape[1], 1)
    traj = np.concatenate([traj, y_axis[:, :, None]], axis=-1)

    x = x.cpu().numpy()

    if np.unique(x.flatten()).shape[0] == 2:
        for i in range(traj.shape[1]):
            if x[i] == -1:
                plt.plot(traj[:, i, 0],
                         traj[:, i, 1],
                         alpha=0.5,
                         c='#F4A582',
                         linewidth=0.5,
                         zorder=0)
                plt.scatter(traj[0, i, 0],
                            traj[0, i, 1],
                            s=4,
                            alpha=1,
                            c='#B2182B',
                            marker='x',
                            zorder=10,
                            linewidths=0.5)
                plt.scatter(traj[-1, i, 0],
                            traj[-1, i, 1],
                            s=4,
                            alpha=1,
                            c='#B2182B',
                            zorder=10,
                            linewidths=0.5)
            else:
                plt.plot(traj[:, i, 0],
                         traj[:, i, 1],
                         alpha=0.5,
                         c='#92C5DE',
                         linewidth=0.5,
                         zorder=0)
                plt.scatter(traj[0, i, 0],
                            traj[0, i, 1],
                            s=4,
                            alpha=1,
                            c='#2166AC',
                            marker='x',
                            zorder=10,
                            linewidths=0.5)
                plt.scatter(traj[-1, i, 0],
                            traj[-1, i, 1],
                            s=4,
                            alpha=1,
                            c='#2166AC',
                            zorder=10,
                            linewidths=0.5)
    else:
        plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.7, alpha=0.1, c='olive', linewidth=0.1)
        plt.scatter(traj[0, :n, 0],
                    traj[0, :n, 1],
                    s=16,
                    alpha=0.56,
                    c=x[:n],
                    marker='x',
                    cmap=my_colormap)
        plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=8, alpha=1, c=x[:n], cmap=my_colormap)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, pad_inches=0, bbox_inches='tight')
    plt.close()


def create_gif(all_traj, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()
        traj = all_traj[frame]
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.scatter(traj[0, :, 0], traj[0, :, 1], s=10, alpha=0.8, c='black')
        ax.scatter(traj[:, :, 0], traj[:, :, 1], s=0.2, alpha=0.2, c='#FFEE99')
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=4, alpha=1, c='blue')
        ax.set_xticks([])
        ax.set_yticks([])

    anim = FuncAnimation(fig, update, frames=len(all_traj), repeat=True)
    anim.save(save_path, writer=PillowWriter(fps=10))
    plt.close(fig)


class MLP(nn.Module):

    def __init__(self, dim: int, expansion_ratio: float = 4.0):
        super().__init__()

        hidden_dim = int(expansion_ratio * dim)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class SimpleNet(nn.Module):

    def __init__(self, in_dim: int, dim: int, depth: int):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, dim)
        self.time_emb = nn.Linear(1, dim)

        self.blocks = nn.ModuleList([MLP(dim) for _ in range(depth)])

        self.output_proj = nn.Linear(dim, in_dim)

    def forward(self, x, t):
        x = self.input_proj(x)
        t = self.time_emb(t)

        x = x + t
        for block in self.blocks:
            x = block(x) + x

        x = self.output_proj(x)

        return x


class ConditionalNet(nn.Module):

    def __init__(self, in_dim: int, dim: int, depth: int):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, dim)
        self.cond_proj = nn.Linear(1, dim)
        self.time_emb = nn.Linear(1, dim)

        self.blocks = nn.ModuleList([MLP(dim) for _ in range(depth)])

        self.output_proj = nn.Linear(dim, in_dim)

    def forward(self, x, c, t):
        x = self.input_proj(x)
        c = self.cond_proj(c)
        t = self.time_emb(t)

        x = x + c + t
        for block in self.blocks:
            x = block(x) + x

        x = self.output_proj(x)

        return x
