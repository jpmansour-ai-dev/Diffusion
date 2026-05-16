import torch

T    = 1000
beta = torch.linspace(1e-4, 2e-2, T)
alpha = 1 - beta
ab    = torch.cumprod(alpha, dim=0)   # ᾱ = ∏ₛ₌₁ᵗ αₛ


def forward_diffusion(
        x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None = None
                      ) -> torch.Tensor:
    """x_t = √ᾱₜ · x_0 + √(1-ᾱₜ) · ε"""
    assert t.shape == (x_0.shape[0],)
    if eps is None:
        eps = torch.randn_like(x_0)
    ab_t = ab[t].view(-1, 1, 1, 1).to(x_0.device)
    return ab_t.sqrt() * x_0 + (1 - ab_t).sqrt() * eps
