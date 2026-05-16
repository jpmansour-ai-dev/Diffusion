import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion.forward import T, forward_diffusion


def fit(model, loader: DataLoader, epochs: int = 5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    device    = next(model.parameters()).device
    model.train()

    losses = []

    for epoch in range(epochs):
        running, steps = 0.0, 0

        for x_0, _ in loader:
            x_0 = x_0.to(device)
            t   = torch.randint(0, T, (x_0.shape[0],))   # CPU — indexes schedule tensors
            eps = torch.randn_like(x_0)

            x_t  = forward_diffusion(x_0, t, eps)
            loss = F.mse_loss(model(x_t, t.to(device)), eps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            steps   += 1

        avg = running / steps
        losses.append(avg)
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg:.4f}")

    return losses
