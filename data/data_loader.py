import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CELEBA_DATASET = "flwrlabs/celeba"
IMAGE_SIZE = 64


def build_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Resize CelebA images and normalize RGB channels to [-1, 1]."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


class CelebADataset(Dataset):
    """CelebA split as tensors ready for diffusion training."""

    def __init__(self, split: str = "train", image_size: int = IMAGE_SIZE) -> None:
        self.dataset = load_dataset(CELEBA_DATASET, split=split)
        self.transform = build_transform(image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        return self.transform(image), 0


def get_celeba_loader(
    batch_size: int = 8,
    num_workers: int = 2,
    split: str = "train",
    image_size: int = IMAGE_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    dataset = CelebADataset(split=split, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
