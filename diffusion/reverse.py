import torch

from diffusion.forward import T, ab, alpha, beta


ab_prev = torch.cat([torch.ones(1), ab[:-1]])
beta_tilde = beta * (1 - ab_prev) / (1 - ab)
sqrt_beta_tilde = beta_tilde.sqrt()
sqrt_recip_alpha = (1 / alpha).sqrt()
sqrt_one_minus_ab = (1 - ab).sqrt()


def _validate_shape(shape: tuple[int, int, int, int]) -> None:
    if not isinstance(shape, tuple) or len(shape) != 4:
        raise ValueError("shape must be a tuple (batch, channels, height, width)")
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"all shape dimensions must be positive, got {shape}")


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def reverse_step(model, x_t: torch.Tensor, t: int) -> torch.Tensor:
    """Draw one sample from p_theta(x_{t-1} | x_t)."""
    if not 0 <= t < T:
        raise ValueError(f"t must be in [0, {T - 1}], got {t}")

    t_batch = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
    eps = model(x_t, t_batch)
    if eps.shape != x_t.shape:
        raise ValueError(
            f"model output must match x_t shape {tuple(x_t.shape)}, got {tuple(eps.shape)}"
        )

    mu = sqrt_recip_alpha[t].item() * (
        x_t - beta[t].item() / sqrt_one_minus_ab[t].item() * eps
    )

    if t == 0:
        return mu
    return mu + sqrt_beta_tilde[t].item() * torch.randn_like(x_t)


@torch.no_grad()
def sample(
    model,
    shape: tuple[int, int, int, int],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate x_0 by starting from Gaussian noise and denoising for T steps."""
    _validate_shape(shape)
    device = torch.device(device) if device is not None else _model_device(model)
    x = torch.randn(shape, device=device)
    for t in reversed(range(T)):
        x = reverse_step(model, x, t)
    return x
