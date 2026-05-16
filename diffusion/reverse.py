import torch
from diffusion.forward import T, beta, alpha, ab

ab_prev = torch.cat([torch.ones(1), ab[:-1]])
beta_tilde = beta * (1 - ab_prev) / (1 - ab)
sqrt_beta_tilde = beta_tilde.sqrt()
sqrt_recip_alpha = (1 / alpha).sqrt()
sqrt_one_minus_ab = (1 - ab).sqrt()


def _validate_shape(shape: tuple) -> None:
    assert isinstance(shape, (tuple, list)) and len(shape) == 4


def images_to_uint8_grid(x: torch.Tensor, nrow: int | None = None):
    """Convert a batch in [-1, 1] to a uint8 image grid."""
    import numpy as np

    if nrow is None:
        nrow = x.shape[0]

    imgs = (x.detach().clamp(-1, 1) + 1) / 2
    rows = []
    for start in range(0, imgs.shape[0], nrow):
        rows.append(torch.cat([img for img in imgs[start:start + nrow]], dim=2))
    grid = torch.cat(rows, dim=1)
    return (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")


@torch.no_grad()
def reverse_step(model, x_t: torch.Tensor, t: int) -> torch.Tensor:
    """One denoising step: p_theta(x_{t-1} | x_t)."""
    eps = model(x_t, torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device))
    assert eps.shape == x_t.shape

    mu = sqrt_recip_alpha[t].item() * (
        x_t - beta[t].item() / sqrt_one_minus_ab[t].item() * eps
    )

    if t == 0:
        return mu
    return mu + sqrt_beta_tilde[t].item() * torch.randn_like(x_t)


@torch.no_grad()
def sample(model, shape: tuple, device: torch.device) -> torch.Tensor:
    """Full reverse diffusion: x_T ~ N(0, I) to x_0."""
    _validate_shape(shape)
    x = torch.randn(shape, device=device)
    for t in reversed(range(T)):
        x = reverse_step(model, x, t)
    return x


@torch.no_grad()
def sample_to_gif(
    model,
    shape: tuple,
    device: torch.device,
    gif_path: str = "denoising.gif",
    capture_every: int = 50,
    restart_pause: int = 8,
) -> None:
    """Save a looping GIF that runs from noise to clean, then restarts."""
    from PIL import Image

    _validate_shape(shape)
    x = torch.randn(shape, device=device)
    frames = []

    def to_frame(batch):
        return Image.fromarray(images_to_uint8_grid(batch, nrow=shape[0]))

    for t in reversed(range(T)):
        if t % capture_every == 0:
            frames.append(to_frame(x))
        x = reverse_step(model, x, t)

    final_frame = to_frame(x)
    frames.append(final_frame)
    frames.extend([final_frame.copy() for _ in range(restart_pause)])

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )


@torch.no_grad()
def sample_progressive(
    model,
    shape: tuple,
    device: torch.device,
    timesteps_to_show: list | None = None,
) -> dict:
    """Return snapshots at selected reverse-process noise levels.

    t=1000 is the initial pure noise sample for a 1000-step schedule.
    t=0 is the final generated sample.
    """
    _validate_shape(shape)
    if timesteps_to_show is None:
        timesteps_to_show = [500, 400, 300, 200, 100, 0]

    timesteps_to_show = sorted(set(timesteps_to_show), reverse=True)
    x = torch.randn(shape, device=device)
    snapshots = {}

    if T in timesteps_to_show:
        snapshots[T] = x.clone()

    for t in reversed(range(T)):
        if (t + 1) in timesteps_to_show:
            snapshots[t + 1] = x.clone()
        x = reverse_step(model, x, t)

    if 0 in timesteps_to_show:
        snapshots[0] = x.clone()

    return snapshots
