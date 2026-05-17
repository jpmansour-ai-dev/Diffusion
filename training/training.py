import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion.forward import T, forward_diffusion


def fit(
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 2e-4,
    max_batches: int | None = None,
    grad_clip: float = 1.0,
    device: torch.device | str | None = None,
) -> list[float]:
    """Train the noise predictor with the standard DDPM epsilon objective."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    losses = []
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{model.__class__.__name__} parameters: {parameters:,}")

    for epoch in range(epochs):
        running, steps = 0.0, 0

        for batch_idx, (x_0, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x_0 = x_0.to(device)
            t = torch.randint(0, T, (x_0.shape[0],), device=device)
            eps = torch.randn_like(x_0)

            x_t = forward_diffusion(x_0, t, eps)
            loss = F.mse_loss(model(x_t, t), eps)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running += loss.item()
            steps += 1

        avg = running / max(steps, 1)
        losses.append(avg)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg:.4f}")

    return losses
