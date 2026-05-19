import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


celeba_dataset_path = "flwrlabs/celeba"
resolution = 64


def preprocess(image_size: int = resolution) -> transforms.Compose:
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
    """Loads and preprocesses CelebA images for model training."""
    def __init__(self, split: str = "train", image_size: int = resolution) -> None:
        self.dataset = load_dataset(celeba_dataset_path, split=split)
        self.transform = preprocess(image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, sample_index):
        return self.transform(self.dataset[sample_index]["image"].convert("RGB"))


def get_celeba_loader(batch_size: int = 8) -> DataLoader:
    """Returns a DataLoader of preprocessed CelebA images."""
    dataset = CelebADataset(split="train", image_size=resolution)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )








