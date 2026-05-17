import torch


T = 1000
beta = torch.linspace(1e-4, 2e-2, T)
alpha = 1 - beta
ab = torch.cumprod(alpha, dim=0)


def forward_diffusion(
    x_0: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample q(x_t | x_0) using the closed-form DDPM noising equation."""
    if t.shape != (x_0.shape[0],):
        raise ValueError(f"t must have shape ({x_0.shape[0]},), got {tuple(t.shape)}")

    if eps is None:
        eps = torch.randn_like(x_0)
    if eps.shape != x_0.shape:
        raise ValueError(
            f"eps must match x_0 shape {tuple(x_0.shape)}, got {tuple(eps.shape)}"
        )

    alpha_bar = ab.to(device=x_0.device, dtype=x_0.dtype)
    alpha_bar_t = alpha_bar[t.to(device=x_0.device, dtype=torch.long)].view(
        -1, 1, 1, 1
    )
    return alpha_bar_t.sqrt() * x_0 + (1 - alpha_bar_t).sqrt() * eps
