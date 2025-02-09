import torch
import yaml

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