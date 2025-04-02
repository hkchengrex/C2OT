from pathlib import Path

import torch
from torchvision.utils import save_image


def euler_integrate(model, num_steps: int, x: torch.Tensor, y: torch.Tensor):
    steps = torch.linspace(0, 1, num_steps + 1, device=x.device)
    for ti, t in enumerate(steps[:-1]):
        flow = model(t.unsqueeze(0), x, y)
        next_t = steps[ti + 1]
        dt = next_t - t
        x = x + dt * flow
    return x


def generate_samples(model,
                     savedir: Path,
                     it: int,
                     conditional: bool = True,
                     num_samples: int = 64,
                     *,
                     num_classes: int = None,
                     size: int,
                     num_steps: int = 100,
                     class_features: dict[str, torch.Tensor] = None):
    device = next(model.parameters()).device
    x0 = torch.randn(num_samples, 3, size, size, device=device)
    x = x0

    if conditional:
        if num_classes and num_classes < num_samples:
            y = torch.arange(num_classes,
                             device=device).repeat_interleave(num_samples // num_classes + 1)
            y = y[:x0.shape[0]]
        else:
            y = torch.arange(num_samples, device=device)
        if class_features is not None:
            # pick first num_samples keys
            file_names = list(class_features.keys())[:num_samples]
            y = torch.stack([class_features[fn] for fn in file_names])
    else:
        y = None

    x = euler_integrate(model, num_steps, x, y)
    x = x.view([-1, 3, size, size]).clip(-1, 1)
    x = x / 2 + 0.5
    savedir.mkdir(parents=True, exist_ok=True)
    save_image(x, savedir / f'{it:09d}.png', nrow=8)


@torch.inference_mode()
def ema_update(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay + source_dict[key].data *
                                    (1 - decay))
