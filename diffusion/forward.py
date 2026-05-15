
import torch

# Noise schedule
T              = 1000
beta_0         = 0.0001  # added noise variance at t=0
beta_T_minus_1 = 0.02    # added noise variance at t=T-1
beta_t         = torch.linspace(beta_0, beta_T_minus_1, T)  # Linear interpolation; shape (T,)
alphas_cumprod = torch.cumprod(1 - beta_t, dim=0) # ᾱₜ = ∏_{i=0}^{t} (1 - β_i) 


# Forward diffusion process:
def forward_diffusion(
        x_0: torch.Tensor, timestep: torch.Tensor, epsilon:  torch.Tensor | None = None
        ) -> torch.Tensor:
    """
    Forward equation:  x_t = √ᾱₜ · x_0  +  √(1 - ᾱₜ) · ε

    x_0:      (B, C, H, W) — original clean images
    timestep: (B,)         — per-sample timestep in [0, T)
    epsilon:  (B, C, H, W) — noise to add; sampled from N(0, I) if None
    Returns:  (B, C, H, W) — noised images x_t
    """
    if epsilon is None:
        epsilon = torch.randn_like(x_0)

    # Reshape (B,) → (B, 1, 1, 1) so alpha_bar broadcasts over (C, H, W)
    alpha_bar = alphas_cumprod[timestep].view(-1, 1, 1, 1)
    x_t = alpha_bar.sqrt() * x_0 + (1 - alpha_bar).sqrt() * epsilon
    return x_t
