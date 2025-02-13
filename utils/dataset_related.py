import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Read CSV file
        df = pd.read_csv(label_csv)
        self.image_filenames = df["image"].tolist()
        self.labels = df["label"].tolist()

        # Convert label names to indices
        unique_labels = sorted(set(self.labels))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = [self.label_to_index[label] for label in self.labels]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
