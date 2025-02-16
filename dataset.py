from torch.utils.data import Dataset
from PIL import Image
import os


class IncrementalDataset(Dataset):
    def __init__(self, root_dir, class_range, transform=None):
        self.root_dir = root_dir
        self.class_range = class_range
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        class_folders = sorted(os.listdir(self.root_dir))
        for class_idx in self.class_range:
            if class_idx < len(class_folders):
                class_folder = class_folders[class_idx]
                class_path = os.path.join(self.root_dir, class_folder)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        samples.append((img_path, class_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label