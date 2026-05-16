import torchvision.transforms as T
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


def get_celeba_loader(root: str = "data/", batch_size: int = 8, num_workers: int = 2) -> DataLoader:
    transform = T.Compose([
        T.Resize(64),
        T.CenterCrop(64),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # → [-1, 1]
    ])
    ds = CelebA(root=root, split="train", transform=transform, download=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
