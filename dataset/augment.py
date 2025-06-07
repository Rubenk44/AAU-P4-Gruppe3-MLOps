import os
import pickle
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import shutil

input_dir = "dataset/original_data/cifar-10-batches-py"
output_dir = "dataset/augmented_data/cifar-10-batches-py"

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


# Process each training batch
for fname in tqdm(os.listdir(input_dir)):
    input_path = os.path.join(input_dir, fname)

    # Only augment the data batches, not meta/test
    if fname.startswith("data_batch"):
        data_dict = load_batch(input_path)
        images = data_dict[b'data']
        labels = data_dict[b'labels']

        # Reshape and augment
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
            "data": augmented,
            "labels": labels,
            "filenames": [
                fn.decode() if isinstance(fn, bytes) else fn
                for fn in data_dict.get(b'filenames', [])
            ],
            "batch_label": data_dict.get(b'batch_label', b"augmented").decode(),
        }

        save_batch(new_batch, os.path.join(output_dir, fname))

    elif fname in ["batches.meta", "test_batch"]:
        # Just copy unmodified
        shutil.copy(input_path, os.path.join(output_dir, fname))
