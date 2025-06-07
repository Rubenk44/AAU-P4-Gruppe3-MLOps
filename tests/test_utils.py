import os
import torch
import yaml
from unittest import mock
from unittest.mock import mock_open
import pytest
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
    config = {'train': {'optimizer': 'adam', 'lr': 0.002}}
    optimizer = pick_optimizer(model, config)
    assert isinstance(optimizer, Adam)
    assert optimizer.defaults['lr'] == 0.002


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


# Additional comprehensive tests for better coverage


def test_load_config_file_not_found():
    """Test load_config with non-existent file"""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yaml")


def test_load_config_invalid_yaml(tmp_path):
    """Test load_config with invalid YAML content"""
    config_path = tmp_path / "invalid_config.yaml"
    with open(config_path, "w") as f:
        f.write("invalid: yaml: content: [")

    with pytest.raises(yaml.YAMLError):
        load_config(config_path)


def test_load_config_empty_file(tmp_path):
    """Test load_config with empty file"""
    config_path = tmp_path / "empty_config.yaml"
    config_path.touch()  # Create empty file

    with mock.patch('builtins.print') as mock_print:
        config = load_config(config_path)
        mock_print.assert_called_with("Config loaded from file:", "Unknown Dataset")
    assert config is None


def test_load_config_with_dataset_name(tmp_path):
    """Test load_config prints correct dataset name"""
    config_content = {'dataset': 'MNIST'}
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    with mock.patch('builtins.print') as mock_print:
        load_config(config_path)
        mock_print.assert_called_with("Config loaded from file:", "MNIST")


def test_load_config_without_dataset_name(tmp_path):
    """Test load_config with config that has no dataset key"""
    config_content = {'model': 'resnet'}
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    with mock.patch('builtins.print') as mock_print:
        load_config(config_path)
        mock_print.assert_called_with("Config loaded from file:", "Unknown Dataset")


def test_pick_optimizer_missing_train_key():
    """Test pick_optimizer with missing 'train' key"""
    model = torch.nn.Linear(10, 2)
    config = {'optimizer': 'adam', 'lr': 0.001}

    with pytest.raises(KeyError):
        pick_optimizer(model, config)


def test_pick_optimizer_missing_optimizer_key():
    """Test pick_optimizer with missing 'optimizer' key"""
    model = torch.nn.Linear(10, 2)
    config = {'train': {'lr': 0.001}}

    with pytest.raises(KeyError):
        pick_optimizer(model, config)


def test_pick_optimizer_unsupported_optimizer():
    """Test pick_optimizer with unsupported optimizer type"""
    model = torch.nn.Linear(10, 2)
    config = {'train': {'optimizer': 'sgd', 'lr': 0.001}}

    # This should raise UnboundLocalError since optimizer is not defined for 'sgd'
    with pytest.raises(UnboundLocalError):
        pick_optimizer(model, config)


def test_pick_optimizer_missing_lr():
    """Test pick_optimizer with missing learning rate"""
    model = torch.nn.Linear(10, 2)
    config = {'train': {'optimizer': 'adam'}}

    with pytest.raises(KeyError):
        pick_optimizer(model, config)


def test_pick_scheduler_missing_train_key():
    """Test pick_scheduler with missing 'train' key"""
    model = torch.nn.Linear(10, 2)
    config = {'scheduler': 'steplr', 'step_size': 10, 'gamma': 0.1}
    optimizer = Adam(model.parameters(), lr=0.001)

    with pytest.raises(KeyError):
        pick_scheduler(optimizer, config)


def test_pick_scheduler_missing_scheduler_key():
    """Test pick_scheduler with missing 'scheduler' key"""
    model = torch.nn.Linear(10, 2)
    config = {'train': {'step_size': 10, 'gamma': 0.1}}
    optimizer = Adam(model.parameters(), lr=0.001)

    with pytest.raises(KeyError):
        pick_scheduler(optimizer, config)


def test_pick_scheduler_unsupported_scheduler():
    """Test pick_scheduler with unsupported scheduler type"""
    model = torch.nn.Linear(10, 2)
    config = {'train': {'scheduler': 'cosine', 'step_size': 10, 'gamma': 0.1}}
    optimizer = Adam(model.parameters(), lr=0.001)

    # This should raise UnboundLocalError since scheduler is not defined for 'cosine'
    with pytest.raises(UnboundLocalError):
        pick_scheduler(optimizer, config)


def test_pick_scheduler_missing_parameters():
    """Test pick_scheduler with missing required parameters"""
    model = torch.nn.Linear(10, 2)
    config = {'train': {'scheduler': 'steplr'}}
    optimizer = Adam(model.parameters(), lr=0.001)

    with pytest.raises(KeyError):
        pick_scheduler(optimizer, config)


@mock.patch("os.path.exists")
@mock.patch("os.listdir")
@mock.patch("os.path.isdir")
@mock.patch("wandb.login")
@mock.patch("wandb.init")
def test_begin_wandb_no_wandb_dir(
    mock_init, mock_login, mock_isdir, mock_listdir, mock_exists
):
    """Test begin_wandb when wandb directory doesn't exist"""
    mock_exists.return_value = False

    begin_wandb()

    mock_login.assert_called_once()
    mock_init.assert_called_once_with(project='MLOps', name="Job 0")


@mock.patch("os.path.exists")
@mock.patch("os.listdir")
@mock.patch("os.path.isdir")
@mock.patch("wandb.login")
@mock.patch("wandb.init")
def test_begin_wandb_with_existing_runs(
    mock_init, mock_login, mock_isdir, mock_listdir, mock_exists
):
    """Test begin_wandb when wandb directory exists with some runs"""
    mock_exists.return_value = True
    mock_listdir.return_value = ["run1", "run2", "file.txt"]
    mock_isdir.side_effect = lambda path: "run" in os.path.basename(path)

    begin_wandb()

    mock_login.assert_called_once()
    mock_init.assert_called_once_with(project='MLOps', name="Job 2")


@mock.patch("os.path.exists")
@mock.patch("os.listdir")
@mock.patch("wandb.login")
@mock.patch("wandb.init")
def test_begin_wandb_empty_directory(mock_init, mock_login, mock_listdir, mock_exists):
    """Test begin_wandb when wandb directory exists but is empty"""
    mock_exists.return_value = True
    mock_listdir.return_value = []

    begin_wandb()

    mock_login.assert_called_once()
    mock_init.assert_called_once_with(project='MLOps', name="Job 0")


@mock.patch("wandb.login")
@mock.patch("wandb.init")
def test_begin_wandb_login_failure(mock_init, mock_login):
    """Test begin_wandb when wandb login fails"""
    mock_login.side_effect = Exception("Login failed")

    with pytest.raises(Exception, match="Login failed"):
        begin_wandb()


@mock.patch("builtins.open", new_callable=mock_open)
@mock.patch("json.dump")
@mock.patch("torch.onnx.export")
@mock.patch("torch.randn")
def test_model_export_detailed(mock_randn, mock_onnx_export, mock_json_dump, mock_file):
    """Test model_export with detailed parameter checking"""
    model = torch.nn.Linear(10, 2)
    device = torch.device("cpu")
    config = {'train': {'optimizer': 'adam', 'lr': 0.001}}
    dummy_tensor = torch.randn(1, 3, 32, 32)
    mock_randn.return_value = dummy_tensor

    with mock.patch('builtins.print') as mock_print:
        model_export(model, device, config)

    # Check that config was written to the correct file
    mock_json_dump.assert_called_once_with(
        config, mock_file.return_value.__enter__.return_value, indent=4
    )

    # Check that dummy input was generated
    mock_randn.assert_called_with(1, 3, 32, 32)

    # Check ONNX export call
    mock_onnx_export.assert_called_once_with(
        model,
        dummy_tensor.to(device),
        "models/Model_latest.onnx",
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Check print message
    mock_print.assert_called_with(
        "Model & config file saved as: models/Model_latest.onnx "
        "& models/Config_latest.json"
    )


@mock.patch("torch.onnx.export")
def test_model_export_with_different_device(mock_onnx_export):
    """Test model_export with different device"""
    model = torch.nn.Linear(10, 2)
    device = torch.device("cpu")
    config = {'train': {'optimizer': 'adam', 'lr': 0.001}}

    with mock.patch("builtins.open", mock_open()):
        with mock.patch("json.dump"):
            model_export(model, device, config)

    # Verify that ONNX export was called
    mock_onnx_export.assert_called_once()

    # Verify that the dummy input is passed to the export function
    args, kwargs = mock_onnx_export.call_args
    model_arg = args[0]
    dummy_input = args[1]

    # Check that model and dummy input are passed
    assert model_arg is model
    assert dummy_input.shape == torch.Size([1, 3, 32, 32])
    assert dummy_input.device.type == device.type


@mock.patch("torch.backends.mps.is_available")
@mock.patch("torch.cuda.is_available")
def test_device_conf_mps_available(mock_cuda_available, mock_mps_available):
    """Test device_conf when MPS is available"""
    mock_mps_available.return_value = True
    mock_cuda_available.return_value = False

    with mock.patch('builtins.print') as mock_print:
        device = device_conf()

    assert device == torch.device("mps")
    mock_print.assert_called_with("Using Apple Silicon GPU with MPS")


@mock.patch("torch.backends.mps.is_available")
@mock.patch("torch.cuda.is_available")
def test_device_conf_cuda_available(mock_cuda_available, mock_mps_available):
    """Test device_conf when CUDA is available but MPS is not"""
    mock_mps_available.return_value = False
    mock_cuda_available.return_value = True

    with mock.patch('builtins.print') as mock_print:
        device = device_conf()

    assert device == torch.device("cuda")
    mock_print.assert_called_with("Using CUDA GPU")


@mock.patch("torch.backends.mps.is_available")
@mock.patch("torch.cuda.is_available")
def test_device_conf_cpu_fallback(mock_cuda_available, mock_mps_available):
    """Test device_conf when neither MPS nor CUDA is available"""
    mock_mps_available.return_value = False
    mock_cuda_available.return_value = False

    with mock.patch('builtins.print') as mock_print:
        device = device_conf()

    assert device == torch.device("cpu")
    mock_print.assert_called_with("Using CPU")


def test_pick_optimizer_with_different_lr_values():
    """Test pick_optimizer with various learning rate values"""
    model = torch.nn.Linear(10, 2)

    # Test with very small learning rate
    config = {'train': {'optimizer': 'adam', 'lr': 1e-6}}
    optimizer = pick_optimizer(model, config)
    assert isinstance(optimizer, Adam)
    assert optimizer.defaults['lr'] == 1e-6

    # Test with large learning rate
    config = {'train': {'optimizer': 'adam', 'lr': 1.0}}
    optimizer = pick_optimizer(model, config)
    assert isinstance(optimizer, Adam)
    assert optimizer.defaults['lr'] == 1.0


def test_pick_scheduler_with_different_parameters():
    """Test pick_scheduler with various parameter values"""
    model = torch.nn.Linear(10, 2)
    optimizer = Adam(model.parameters(), lr=0.001)

    # Test with different step_size and gamma values
    config = {'train': {'scheduler': 'steplr', 'step_size': 20, 'gamma': 0.9}}
    scheduler = pick_scheduler(optimizer, config)
    assert isinstance(scheduler, StepLR)
    assert scheduler.step_size == 20
    assert scheduler.gamma == 0.9

    # Test with step_size = 1
    config = {'train': {'scheduler': 'steplr', 'step_size': 1, 'gamma': 0.5}}
    scheduler = pick_scheduler(optimizer, config)
    assert isinstance(scheduler, StepLR)
    assert scheduler.step_size == 1
    assert scheduler.gamma == 0.5
