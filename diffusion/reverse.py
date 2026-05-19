import torch

from diffusion.forward import T, alpha_bar, alpha, beta


alpha_bar_prev        = torch.cat([torch.ones(1), alpha_bar[:-1]])
beta_tilde            = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
sqrt_beta_tilde       = beta_tilde.sqrt()
sqrt_recip_alpha      = (1 / alpha).sqrt()
sqrt_one_minus_alpha_bar = (1 - alpha_bar).sqrt()


@torch.no_grad()
def reverse_step(model, x_t: torch.Tensor, t: int) -> torch.Tensor:
    if not 0 <= t < T:
        raise ValueError(f"t must be in [0, {T - 1}], got {t}")

    t_batch = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
    eps = model(x_t, t_batch)

    mu = sqrt_recip_alpha[t].item() * (
        x_t - beta[t].item() / sqrt_one_minus_alpha_bar[t].item() * eps
    )

    if t == 0:
        return mu
    return mu + sqrt_beta_tilde[t].item() * torch.randn_like(x_t)


@torch.no_grad()
def sample(model, shape: tuple[int, int, int, int]) -> torch.Tensor:
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device)
    for t in reversed(range(T)):
        x = reverse_step(model, x, t)
    return x
