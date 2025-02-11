import torch
import yaml
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import wandb
import os
from datetime import datetime
import json


def device_conf():
    # Device configuration (MPS for Mac, CUDA for other GPUs, CPU fallback)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU with MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Config loaded from file:", config.get("dataset", "Unknown Dataset"))
    return config


def add_transform(config, train_subset):
    # missing centercrop from configuration file
    train_trans = []
    list_mean = []
    list_std = []

    for tensor, _ in train_subset:
        list_mean.append(tensor)
        list_std.append(tensor)

    # Calculating std and mean using [channel, height, width]
    mean = torch.stack(list_mean).mean(dim=[0, 2, 3])
    std = torch.stack(list_std).std(dim=[0, 2, 3])

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

    if config['data_augmentation']['rotation'] != (0, 0):
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

    print(f"Following transformations have been added:{train_trans}")

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(config['data_augmentation']['resize'])),
            *train_trans,
            transforms.Normalize(mean, std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Resize(size=(config['data_augmentation']['resize'])),
            transforms.Normalize(mean, std),
        ]
    )

    return train_transform, val_transform


def data_load(config):
    transform = transforms.ToTensor()

    # Downloading dataset
    trainset = torchvision.datasets.CIFAR10(
        root=config['dataset']['data'], train=True, download=True, transform=transform
    )

    # Splitting data into Training and validation
    train_size = int(config['train']['train_test_split'] * len(trainset))
    val_size = len(trainset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        trainset, [train_size, val_size]
    )

    print(train_subset[1][0].shape)
    train_transform, val_transform = add_transform(config, train_subset)
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform
    print(train_subset[1][0].shape)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
    )

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
    )
    return train_loader, val_loader


def pick_optimizer(model, config):
    if config['train']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    return optimizer


def pick_scheduler(optimizer, config):
    if config['train']['scheduler'] == 'steplr':
        scheduler = StepLR(
            optimizer,
            step_size=config['train']['step_size'],
            gamma=config['train']['gamma'],
        )
    return scheduler


def begin_wandb():
    wandb_dir = "wandb"
    if not os.path.exists(wandb_dir):
        runs = []
    else:
        runs = [
            d
            for d in os.listdir(wandb_dir)
            if os.path.isdir(os.path.join(wandb_dir, d))
        ]

    wandb.login()
    wandb.init(project='MLOps', name=f"Job {len(runs)}")


def model_export(model, device, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_filename = f"Model_{timestamp}.onnx"
    config_filename = f"Config_{onnx_filename}.txt"

    with open(config_filename, "w") as f:
        json.dump(config, f, indent=4)

    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Model & config file saved as: {onnx_filename} & {config_filename}")
