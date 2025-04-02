from pathlib import Path

from tensordict import TensorDict
from torch.utils.data.dataset import Dataset


class MemmapImages(Dataset):

    def __init__(self, mmap_dir: Path, transform=None):
        self.transform = transform
        td = TensorDict.load_memmap(mmap_dir)
        self.images = td['image']
        self.labels = td['label']
        self.clip_features = td['clip_features'] if 'clip_features' in td else None

    def __getitem__(self, index):
        image = self.images[index]
        if self.clip_features is None:
            label = self.labels[index]
        else:
            label = self.clip_features[index]

        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)
