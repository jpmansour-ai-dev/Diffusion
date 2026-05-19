import torch


T          = 1000
beta_start = 1e-4
beta_end   = 0.02

beta      = torch.linspace(beta_start, beta_end, T)
alpha     = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)




def forward_diffusion(
    x_0: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample q(x_t | x_0) using the DDPM noising equation."""
    if t.shape != (x_0.shape[0],):
        raise ValueError(f"t must have shape ({x_0.shape[0]},), got {tuple(t.shape)}")

    if eps is None:
        eps = torch.randn_like(x_0)
    if eps.shape != x_0.shape:
        raise ValueError(
            f"epsilon must match x_0 shape {tuple(x_0.shape)}, got {tuple(eps.shape)}"
        )

    alpha_bar_t = alpha_bar.to(device=x_0.device, dtype=x_0.dtype)[
        t.to(device=x_0.device, dtype=torch.long)
    ].view(-1, 1, 1, 1)
    return alpha_bar_t.sqrt() * x_0 + (1 - alpha_bar_t).sqrt() * eps



