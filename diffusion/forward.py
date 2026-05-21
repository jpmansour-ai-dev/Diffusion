import torch


T = 1000
# Defining the cosine schedule:
s = 0.008
f = torch.cos(
    (torch.arange(T + 1, dtype=torch.float64) / T + s) / (1 + s) * math.pi / 2
    ) ** 2
alpha_bar = (f / f[0]).float()[1:]
beta = (1 - alpha_bar / torch.cat([torch.ones(1), alpha_bar[:-1]])).clamp(max=0.999)
alpha = 1 - beta


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



