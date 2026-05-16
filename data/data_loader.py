import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


transform = T.Compose([
    T.Resize(64),
    T.CenterCrop(64),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # → [-1, 1]
])


class CelebADataset(Dataset):

    def __init__(self):
        self.ds = load_dataset("flwrlabs/celeba", split="train")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return transform(self.ds[idx]["image"].convert("RGB")), 0


def get_celeba_loader(batch_size: int = 8, num_workers: int = 2) -> DataLoader:
    return DataLoader(CelebADataset(), batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)
