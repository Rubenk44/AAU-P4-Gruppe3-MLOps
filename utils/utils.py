import torch
import yaml
from torch.optim.lr_scheduler import StepLR
import wandb
import os
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
    if config is None:
        print("Config loaded from file:", "Unknown Dataset")
    else:
        print("Config loaded from file:", config.get("dataset", "Unknown Dataset"))
    return config


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
    model = model.float()
    onnx_filename = config["output"]["model_path"]
    config_filename = config["output"]["config_path"]

    os.makedirs(os.path.dirname(config["output"]["model_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["output"]["config_path"]), exist_ok=True)

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
