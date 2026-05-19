import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion.forward import T, forward_diffusion


def fit(
    model: torch.nn.Module, loader: DataLoader, epochs: int, model_path: str, 
    learning_rate: float = 2e-4, max_grad_norm: float = 1,
) -> list[float]:
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    losses = []
    parameters = sum(p.numel() for p in model.parameters())                        
    print(f"{model.__class__.__name__} U-Net learnable parameters count: {parameters:,}")

    for epoch in range(epochs):
        total_loss, num_batches = 0, 0

        for x_0 in loader:
            x_0 = x_0.to(device)
            t = torch.randint(0, T, (x_0.shape[0],), device=device)
            eps = torch.randn_like(x_0)

            x_t       = forward_diffusion(x_0, t, eps)
            eps_pred  = model(x_t, t)
            loss      = F.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.5f}")
        torch.save(model.state_dict(), model_path)

    return losses


