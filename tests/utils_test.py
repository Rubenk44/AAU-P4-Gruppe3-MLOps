import os

# import folder_up
import torch
import yaml
from unittest import mock
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils.utils import (
    device_conf,
    load_config,
    pick_optimizer,
    pick_scheduler,
    begin_wandb,
    model_export,
)


def test_device_conf():
    device = device_conf()
    assert device in [torch.device("mps"), torch.device("cuda"), torch.device("cpu")]


def test_load_config(tmp_path):
    config_content = {
        'dataset': 'CIFAR-10',
        'train': {
            'optimizer': 'adam',
            'lr': 0.001,
            'scheduler': 'steplr',
            'step_size': 10,
            'gamma': 0.1,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    config = load_config(config_path)
    assert config == config_content


def test_pick_optimizer():
    model = torch.nn.Linear(10, 2)
    config = {'train': {'optimizer': 'adam', 'lr': 0.001}}
    optimizer = pick_optimizer(model, config)
    assert isinstance(optimizer, Adam)
    assert optimizer.defaults['lr'] == 0.001


def test_pick_scheduler():
    model = torch.nn.Linear(10, 2)
    config = {'train': {'scheduler': 'steplr', 'step_size': 10, 'gamma': 0.1}}
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = pick_scheduler(optimizer, config)
    assert isinstance(scheduler, StepLR)
    assert scheduler.step_size == 10
    assert scheduler.gamma == 0.1


@mock.patch("wandb.login")
@mock.patch("wandb.init")
def test_begin_wandb(mock_init, mock_login, tmp_path):
    os.makedirs(tmp_path / "wandb", exist_ok=True)
    begin_wandb()
    mock_login.assert_called_once()
    mock_init.assert_called_once_with(project='MLOps', name=mock.ANY)


@mock.patch("torch.onnx.export")
def test_model_export(mock_onnx_export, tmp_path):
    model = torch.nn.Linear(10, 2)
    device = torch.device("cpu")
    config = {'train': {'optimizer': 'adam', 'lr': 0.001}}

    model_export(model, device, config)
    mock_onnx_export.assert_called_once()
