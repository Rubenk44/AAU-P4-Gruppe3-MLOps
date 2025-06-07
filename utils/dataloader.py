import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os
import subprocess
import torch.distributed as dist


def is_rank_0():
    return not dist.is_initialized() or dist.get_rank() == 0

def add_transform(config, train_subset):
    # missing centercrop from configuration file
    train_trans = []
    list_mean = []
    list_std = []
    c, h, w = train_subset[1][0].shape

    for tensor, _ in train_subset:
        list_mean.append(tensor)
        list_std.append(tensor)

    # Calculating std and mean using [channel, height, width]
    mean = torch.stack(list_mean).mean(dim=[0, 2, 3])
    std = torch.stack(list_std).std(dim=[0, 2, 3])

    if (
        config['data_augmentation']['resize'][0] < 32
        or config['data_augmentation']['resize'][1] < 32
    ):
        if is_rank_0():
            print("No resizing since minimum is 32x32")
    elif config['data_augmentation']['resize'] != [h, w]:
        train_trans.append(
            transforms.Resize(size=(config['data_augmentation']['resize']))
        )

    if config['data_augmentation']['horizontal_flip'] > 0:
        train_trans.append(
            transforms.RandomHorizontalFlip(
                config['data_augmentation']['horizontal_flip']
            )
        )

    if config['data_augmentation']['vertical_flip'] > 0:
        train_trans.append(
            transforms.RandomVerticalFlip(config['data_augmentation']['vertical_flip'])
        )

    if any(
        config['data_augmentation'][i] > 0
        for i in ['brightness', 'contrast', 'saturation', 'hue']
    ):
        train_trans.append(
            transforms.ColorJitter(
                brightness=config['data_augmentation']['brightness'],
                contrast=config['data_augmentation']['contrast'],
                saturation=config['data_augmentation']['saturation'],
                hue=config['data_augmentation']['hue'],
            )
        )

    if config['data_augmentation']['grayscale'] > 0:
        train_trans.append(
            transforms.RandomGrayscale(config['data_augmentation']['grayscale'])
        )

    if config['data_augmentation']['rotation'] != [0, 0]:
        train_trans.append(
            transforms.RandomRotation(config['data_augmentation']['rotation'])
        )

    if (
        config['data_augmentation']['width_shift'] > 0
        or config['data_augmentation']['height_shift'] > 0
    ):
        train_trans.append(
            transforms.RandomAffine(
                degrees=config['data_augmentation']['degrees'],
                translate=(
                    config['data_augmentation']['width_shift'],
                    config['data_augmentation']['height_shift'],
                ),
            )
        )

    if config['data_augmentation']['distortion_scale'] > 0:
        train_trans.append(
            transforms.RandomPerspective(
                distortion_scale=config['data_augmentation']['distortion_scale']
            )
        )

    if is_rank_0():
        print(f"Following transformations have been added:{train_trans}")

    train_transform = transforms.Compose(
        [
            *train_trans,
            transforms.Normalize(mean, std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Normalize(mean, std),
        ]
    )

    return train_transform, val_transform


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


def data_load(config):
    transform = transforms.ToTensor()

    #   print(
    #      "remember to run aws configure in terminal and setup credentials"
    # )  # print kun når der er error
    # Download dataset
    # check for data folder if not make one

    isExist = os.path.exists(config['dataset']['data'])
    if not isExist:
        os.mkdir(config['dataset']['data'])

    if len(os.listdir(config['dataset']['data'])) == 0:
        print("\nEmpty folder. Importing dataset from S3...")
        download_data = [
            "dvc",
            "import-url",
            config['dataset']['s3_path'],
            config['dataset']['data'],
        ]
        print(config['dataset']['s3_path'])
        subprocess.run(download_data, check=True)

    else:
        print("\nDataset exists in folder. Checking for update using DVC...")
        update_data = ["dvc", "update", config['dataset']['dvc_path']]
        print(config['dataset']['dvc_path'])
        subprocess.run(update_data, check=True)

    # Extracts data from dataset with CIFAR10 datastructure
    trainset = torchvision.datasets.CIFAR10(
        root=config['dataset']['data'], train=True, download=False, transform=transform
    )

    # Splitting data into Training and validation
    train_size = int(config['train']['train_test_split'] * len(trainset))
    val_size = len(trainset) - train_size
    train_subset, val_subset = random_split(trainset, [train_size, val_size])

    print(f"train: {train_subset[1][0].shape}")
    print(f"val: {val_subset[1][0].shape}")

    train_transform, val_transform = add_transform(config, train_subset)

    train_subset = TransformSubset(train_subset, train_transform)
    val_subset = TransformSubset(val_subset, val_transform)

    print(f"train: {train_subset[1][0].shape}")
    print(f"val: {val_subset[1][0].shape}")

    train_loader = DataLoader(
        train_subset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
    )
    return train_loader, val_loader