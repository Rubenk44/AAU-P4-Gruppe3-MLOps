import os
import pickle
import numpy as np
from torchvision import transforms
from tqdm import tqdm

input_dir = "dataset/cifar-10-batches-py"
output_dir = "dataset/cifar-10-augmented"

os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]
)


def load_batch(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='bytes')


def save_batch(data_dict, path):
    with open(path, 'wb') as f:
        pickle.dump(data_dict, f)


# Process each batch file
for fname in tqdm(os.listdir(input_dir)):
    if fname.startswith("data_batch"):
        batch_path = os.path.join(input_dir, fname)
        data_dict = load_batch(batch_path)
        images = data_dict[b'data']
        labels = data_dict[b'labels']

        # CIFAR-10: (N, 3072) → (N, 3, 32, 32)
        images = images.reshape(-1, 3, 32, 32)
        augmented = []

        for img in images:
            img = np.transpose(img, (1, 2, 0))  # CHW → HWC
            img = transform(img).numpy()
            img = (img * 255).astype(np.uint8)
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW
            augmented.append(img)

        augmented = np.stack(augmented).reshape(-1, 3072)
        new_batch = {
            b'data': augmented,
            b'labels': labels,
            b'filenames': data_dict.get(b'filenames', []),
            b'batch_label': data_dict.get(b'batch_label', b"augmented"),
        }

        out_path = os.path.join(output_dir, fname)
        save_batch(new_batch, out_path)

print("✅ Augmentation complete.")
