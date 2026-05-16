import torch
import matplotlib.pyplot as plt

from data.data_loader import get_celeba_loader
from model.unet import UNet
from training.training import fit
from diffusion.reverse import sample


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

loader = get_celeba_loader(batch_size=64)

model  = UNet().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

losses = fit(model, loader, epochs=10)

# ── Sample ────────────────────────────────────────────────────────────
model.eval()
imgs = sample(model, shape=(4, 3, 64, 64), device=device)
imgs = (imgs.clamp(-1, 1) + 1) / 2          # [-1, 1] → [0, 1]

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(imgs[i].permute(1, 2, 0).cpu().numpy())
    ax.axis("off")
plt.tight_layout()
plt.savefig("samples.png")
plt.show()
