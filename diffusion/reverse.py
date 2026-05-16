import torch
from diffusion.forward import T, beta, alpha, ab

ab_prev           = torch.cat([torch.ones(1), ab[:-1]])
beta_tilde        = beta * (1 - ab_prev) / (1 - ab)
sqrt_beta_tilde   = beta_tilde.sqrt()
sqrt_recip_alpha  = (1 / alpha).sqrt()
sqrt_one_minus_ab = (1 - ab).sqrt()


@torch.no_grad()
def reverse_step(model, x_t: torch.Tensor, t: int) -> torch.Tensor:
    """p_θ(x_{t-1} | x_t): one denoising step."""
    eps = model(x_t, torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device))
    assert eps.shape == x_t.shape

    mu = sqrt_recip_alpha[t].item() * (x_t - beta[t].item() / sqrt_one_minus_ab[t].item() * eps)

    if t == 0:
        return mu
    return mu + sqrt_beta_tilde[t].item() * torch.randn_like(x_t)


@torch.no_grad()
def sample(model, shape: tuple, device: torch.device) -> torch.Tensor:
    """Full reverse diffusion: x_T ~ N(0,I) → x_0."""
    assert isinstance(shape, (tuple, list)) and len(shape) == 4
    x = torch.randn(shape, device=device)
    for t in reversed(range(T)):
        x = reverse_step(model, x, t)
    return x
