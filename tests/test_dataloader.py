import os
import torch
import torchvision.transforms as transforms
import pytest
from unittest import mock
from unittest.mock import patch
from torch.utils.data import DataLoader
from utils.dataloader import add_transform, TransformSubset, data_load
import torch.nn as nn
from functools import partial


# This allows bypassing DVC operations in CI environments
def is_ci_environment():
    return os.environ.get('CI') == 'true'


class TestAddTransform:
    """Test suite for add_transform function"""

    def create_mock_subset(self, shape=(3, 32, 32)):
        """Create a mock subset with specified tensor shape"""
        mock_tensor = torch.randn(shape)
        mock_subset = [(mock_tensor, 0), (mock_tensor, 1)]
        return mock_subset

    def test_add_transform_no_augmentations(self):
        """Test add_transform with no augmentations enabled"""
        config = {
            'data_augmentation': {
                'resize': [32, 32],
                'horizontal_flip': 0,
                'vertical_flip': 0,
                'brightness': 0,
                'contrast': 0,
                'saturation': 0,
                'hue': 0,
                'grayscale': 0,
                'rotation': [0, 0],
                'width_shift': 0,
                'height_shift': 0,
                'degrees': 0,
                'distortion_scale': 0,
            }
        }
        train_subset = self.create_mock_subset()

        with mock.patch('builtins.print'):
            train_transform, val_transform = add_transform(config, train_subset)

        # Should only have Normalize transform
        assert len(train_transform.transforms) == 1
        assert isinstance(train_transform.transforms[0], transforms.Normalize)
        assert len(val_transform.transforms) == 1
        assert isinstance(val_transform.transforms[0], transforms.Normalize)

    def test_add_transform_with_resize_larger(self):
        """Test add_transform with resize to larger dimensions"""
        config = {
            'data_augmentation': {
                'resize': [64, 64],
                'horizontal_flip': 0,
                'vertical_flip': 0,
                'brightness': 0,
                'contrast': 0,
                'saturation': 0,
                'hue': 0,
                'grayscale': 0,
                'rotation': [0, 0],
                'width_shift': 0,
                'height_shift': 0,
                'degrees': 0,
                'distortion_scale': 0,
            }
        }
        train_subset = self.create_mock_subset()

        with mock.patch('builtins.print'):
            train_transform, val_transform = add_transform(config, train_subset)

        # Should have Resize and Normalize transforms
        assert len(train_transform.transforms) == 2
        assert isinstance(train_transform.transforms[0], transforms.Resize)
        assert isinstance(train_transform.transforms[1], transforms.Normalize)

    def test_add_transform_with_resize_too_small(self):
        """Test add_transform with resize smaller than 32x32"""
        config = {
            'data_augmentation': {
                'resize': [16, 16],
                'horizontal_flip': 0,
                'vertical_flip': 0,
                'brightness': 0,
                'contrast': 0,
                'saturation': 0,
                'hue': 0,
                'grayscale': 0,
                'rotation': [0, 0],
                'width_shift': 0,
                'height_shift': 0,
                'degrees': 0,
                'distortion_scale': 0,
            }
        }
        train_subset = self.create_mock_subset()

        with mock.patch('builtins.print') as mock_print:
            train_transform, val_transform = add_transform(config, train_subset)

        # Should print warning message
        mock_print.assert_any_call("No resizing since minimum is 32x32")
        # Should only have Normalize transform
        assert len(train_transform.transforms) == 1
        assert isinstance(train_transform.transforms[0], transforms.Normalize)

    def test_add_transform_with_horizontal_flip(self):
        """Test add_transform with horizontal flip enabled"""
        config = {
            'data_augmentation': {
                'resize': [32, 32],
                'horizontal_flip': 0.5,
                'vertical_flip': 0,
                'brightness': 0,
                'contrast': 0,
                'saturation': 0,
                'hue': 0,
                'grayscale': 0,
                'rotation': [0, 0],
                'width_shift': 0,
                'height_shift': 0,
                'degrees': 0,
                'distortion_scale': 0,
            }
        }
        train_subset = self.create_mock_subset()

        with mock.patch('builtins.print'):
            train_transform, val_transform = add_transform(config, train_subset)

        # Should have RandomHorizontalFlip and Normalize transforms
        assert len(train_transform.transforms) == 2
        assert isinstance(
            train_transform.transforms[0], transforms.RandomHorizontalFlip
        )
        assert train_transform.transforms[0].p == 0.5
        assert isinstance(train_transform.transforms[1], transforms.Normalize)

    def test_add_transform_with_vertical_flip(self):
        """Test add_transform with vertical flip enabled"""
        config = {
            'data_augmentation': {
                'resize': [32, 32],
                'horizontal_flip': 0,
                'vertical_flip': 0.3,
                'brightness': 0,
                'contrast': 0,
                'saturation': 0,
                'hue': 0,
                'grayscale': 0,
                'rotation': [0, 0],
                'width_shift': 0,
                'height_shift': 0,
                'degrees': 0,
                'distortion_scale': 0,
            }
        }
        train_subset = self.create_mock_subset()

        with mock.patch('builtins.print'):
            train_transform, val_transform = add_transform(config, train_subset)

        # Should have RandomVerticalFlip and Normalize transforms
        assert len(train_transform.transforms) == 2
        assert isinstance(train_transform.transforms[0], transforms.RandomVerticalFlip)
        assert train_transform.transforms[0].p == 0.3
        assert isinstance(train_transform.transforms[1], transforms.Normalize)

    def test_add_transform_with_color_jitter(self):
        """Test add_transform with color jitter enabled"""
        config = {
            'data_augmentation': {
                'resize': [32, 32],
                'horizontal_flip': 0,
                'vertical_flip': 0,
                'brightness': 0.2,
                'contrast': 0.1,
                'saturation': 0.3,
                'hue': 0.05,
                'grayscale': 0,
                'rotation': [0, 0],
                'width_shift': 0,
                'height_shift': 0,
                'degrees': 0,
                'distortion_scale': 0,
            }
        }
        train_subset = self.create_mock_subset()

        with mock.patch('builtins.print'):
            train_transform, val_transform = add_transform(config, train_subset)

        # Should have ColorJitter and Normalize transforms
        assert len(train_transform.transforms) == 2
        assert isinstance(train_transform.transforms[0], transforms.ColorJitter)
        assert isinstance(train_transform.transforms[1], transforms.Normalize)

    def test_add_transform_with_grayscale(self):
        """Test add_transform with grayscale enabled"""
        config = {
            'data_augmentation': {
                'resize': [32, 32],
                'horizontal_flip': 0,
                'vertical_flip': 0,
                'brightness': 0,
                'contrast': 0,
                'saturation': 0,
                'hue': 0,
                'grayscale': 0.1,
                'rotation': [0, 0],
                'width_shift': 0,
                'height_shift': 0,
                'degrees': 0,
                'distortion_scale': 0,
            }
        }
        train_subset = self.create_mock_subset()

        with mock.patch('builtins.print'):
            train_transform, val_transform = add_transform(config, train_subset)

        # Should have RandomGrayscale and Normalize transforms
        assert len(train_transform.transforms) == 2
        assert isinstance(train_transform.transforms[0], transforms.RandomGrayscale)
        assert train_transform.transforms[0].p == 0.1
        assert isinstance(train_transform.transforms[1], transforms.Normalize)

    def test_add_transform_with_rotation(self):
        """Test add_transform with rotation enabled"""
        config = {
            'data_augmentation': {
                'resize': [32, 32],
                'horizontal_flip': 0,
                'vertical_flip': 0,
                'brightness': 0,
                'contrast': 0,
                'saturation': 0,
                'hue': 0,
                'grayscale': 0,
                'rotation': [-30, 30],
                'width_shift': 0,
                'height_shift': 0,
                'degrees': 0,
                'distortion_scale': 0,
            }
        }
        train_subset = self.create_mock_subset()

        with mock.patch('builtins.print'):
            train_transform, val_transform = add_transform(config, train_subset)

        # Should have RandomRotation and Normalize transforms
        assert len(train_transform.transforms) == 2
        assert isinstance(train_transform.transforms[0], transforms.RandomRotation)
        assert isinstance(train_transform.transforms[1], transforms.Normalize)

    def test_add_transform_with_translation(self):
        """Test add_transform with translation enabled"""
        config = {
            'data_augmentation': {
                'resize': [32, 32],
                'horizontal_flip': 0,
                'vertical_flip': 0,
                'brightness': 0,
                'contrast': 0,
                'saturation': 0,
                'hue': 0,
                'grayscale': 0,
                'rotation': [0, 0],
                'width_shift': 0.1,
                'height_shift': 0.2,
                'degrees': 15,
                'distortion_scale': 0,
            }
        }
        train_subset = self.create_mock_subset()

        with mock.patch('builtins.print'):
            train_transform, val_transform = add_transform(config, train_subset)

        # Should have RandomAffine and Normalize transforms
        assert len(train_transform.transforms) == 2
        assert isinstance(train_transform.transforms[0], transforms.RandomAffine)
        assert isinstance(train_transform.transforms[1], transforms.Normalize)

    def test_add_transform_with_perspective(self):
        """Test add_transform with perspective distortion enabled"""
        config = {
            'data_augmentation': {
                'resize': [32, 32],
                'horizontal_flip': 0,
                'vertical_flip': 0,
                'brightness': 0,
                'contrast': 0,
                'saturation': 0,
                'hue': 0,
                'grayscale': 0,
                'rotation': [0, 0],
                'width_shift': 0,
                'height_shift': 0,
                'degrees': 0,
                'distortion_scale': 0.2,
            }
        }
        train_subset = self.create_mock_subset()

        with mock.patch('builtins.print'):
            train_transform, val_transform = add_transform(config, train_subset)

        # Should have RandomPerspective and Normalize transforms
        assert len(train_transform.transforms) == 2
        assert isinstance(train_transform.transforms[0], transforms.RandomPerspective)
        assert train_transform.transforms[0].distortion_scale == 0.2
        assert isinstance(train_transform.transforms[1], transforms.Normalize)

    def test_add_transform_all_augmentations(self):
        """Test add_transform with all augmentations enabled"""
        config = {
            'data_augmentation': {
                'resize': [64, 64],
                'horizontal_flip': 0.5,
                'vertical_flip': 0.3,
                'brightness': 0.2,
                'contrast': 0.1,
                'saturation': 0.3,
                'hue': 0.05,
                'grayscale': 0.1,
                'rotation': [-30, 30],
                'width_shift': 0.1,
                'height_shift': 0.2,
                'degrees': 15,
                'distortion_scale': 0.2,
            }
        }
        train_subset = self.create_mock_subset()

        with mock.patch('builtins.print'):
            train_transform, val_transform = add_transform(config, train_subset)

        # Should have all transforms plus Normalize
        assert len(train_transform.transforms) == 9  # 8 augmentations + Normalize
        assert isinstance(train_transform.transforms[-1], transforms.Normalize)


class TestTransformSubset:
    """Test suite for TransformSubset class"""

    def test_transform_subset_without_transform(self):
        """Test TransformSubset without any transform"""
        mock_data = [(torch.randn(3, 32, 32), 0), (torch.randn(3, 32, 32), 1)]
        subset = TransformSubset(mock_data, None)

        assert len(subset) == 2
        img, label = subset[0]
        assert img.shape == (3, 32, 32)
        assert label == 0

    def test_transform_subset_with_transform(self):
        """Test TransformSubset with a transform applied"""
        mock_data = [(torch.randn(3, 32, 32), 0), (torch.randn(3, 32, 32), 1)]
        transform = transforms.Resize((64, 64))
        subset = TransformSubset(mock_data, transform)

        assert len(subset) == 2
        img, label = subset[0]
        assert img.shape == (3, 64, 64)
        assert label == 0

    def test_transform_subset_length(self):
        """Test TransformSubset __len__ method"""
        mock_data = [(torch.randn(3, 32, 32), i) for i in range(10)]
        subset = TransformSubset(mock_data, None)

        assert len(subset) == 10


def create_mock_subset_for_data_load():
    """Helper function to create mock subsets for data_load tests"""

    class MockSubset:
        def __init__(self, size):
            self.size = size
            self.data = [(torch.randn(3, 32, 32), i % 10) for i in range(size)]

        def __getitem__(self, index):
            if isinstance(index, int):
                if index < self.size:
                    return self.data[index]
                raise IndexError(f"Index {index} out of range {self.size}")
            elif isinstance(index, slice):
                return [self.data[i] for i in range(*index.indices(self.size))]
            else:
                # Handle special case for train_subset[1][0].shape - data_load does this
                if isinstance(index, tuple) or isinstance(index, list):
                    # For train_subset[1][0].shape, etc.
                    inner_idx = index[0]
                    tensor_idx = index[1] if len(index) > 1 else 0
                    if inner_idx < self.size:
                        return self.data[inner_idx][tensor_idx]
                raise TypeError(f"Invalid index type: {type(index)}, value: {index}")

        def __len__(self):
            return self.size

    return MockSubset


def add_one(x):
    return x + 1


@pytest.fixture
def base_config():
    return {
        'dataset': {
            'version': 'original',
            'paths': {
                'original': {'local_path': '/fake/path'},
                'augmented': {'local_path': '/fake/aug'},
            },
        },
        'train': {'train_test_split': 0.8, 'batch_size': 4, 'num_workers': 0},
        'data_augmentation': {
            'resize': [32, 32],
            'horizontal_flip': 0,
            'vertical_flip': 0,
            'brightness': 0,
            'contrast': 0,
            'saturation': 0,
            'hue': 0,
            'grayscale': 0,
            'rotation': [0, 0],
            'width_shift': 0,
            'height_shift': 0,
            'degrees': 0,
            'distortion_scale': 0,
        },
    }


class TestDataLoad:

    @patch('utils.dataloader.CIFAR10')
    @patch('utils.dataloader.random_split')
    @patch('utils.dataloader.os.listdir')
    def test_data_load_original(
        self, mock_listdir, mock_random_split, mock_cifar10, base_config
    ):
        dummy_tensor = torch.randn(3, 32, 32)
        dummy_dataset = [(dummy_tensor, 0)] * 10
        mock_cifar10.return_value = dummy_dataset
        mock_random_split.return_value = (dummy_dataset[:8], dummy_dataset[8:])
        mock_listdir.return_value = ['dummy_file']

        train_loader, val_loader = data_load(base_config)

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

        batch = next(iter(train_loader))
        images, labels = batch
        assert images.shape[0] == base_config['train']['batch_size']
        assert images.ndim == 4
        assert labels.ndim == 1
        assert isinstance(labels[0].item(), int)

    @patch('utils.dataloader.AugmentedCIFAR10')
    @patch('utils.dataloader.random_split')
    @patch('utils.dataloader.os.listdir')
    def test_data_load_augmented(
        self, mock_listdir, mock_random_split, mock_augmented, base_config
    ):
        base_config['dataset']['version'] = 'augmented'
        dummy_tensor = torch.randn(3, 32, 32)
        dummy_dataset = [(dummy_tensor, 1)] * 10
        mock_augmented.return_value = dummy_dataset
        mock_random_split.return_value = (dummy_dataset[:8], dummy_dataset[8:])
        mock_listdir.return_value = ['dummy_file']

        train_loader, val_loader = data_load(base_config)

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        batch = next(iter(val_loader))
        assert batch[0].shape[1:] == (3, 32, 32)

    @patch('utils.dataloader.add_transform')
    @patch('utils.dataloader.CIFAR10')
    @patch('utils.dataloader.random_split')
    @patch('utils.dataloader.os.listdir')
    def test_data_load_calls_add_transform(
        self, mock_listdir, mock_split, mock_cifar, mock_add, base_config
    ):
        dummy_tensor = torch.randn(3, 32, 32)
        dummy_dataset = [(dummy_tensor, 0)] * 10
        mock_cifar.return_value = dummy_dataset
        mock_listdir.return_value = ['dummy_file']
        mock_split.return_value = (dummy_dataset[:8], dummy_dataset[8:])

        mock_add.return_value = (nn.Identity(), nn.Identity())

        data_load(base_config)
        mock_add.assert_called_once()

    @patch('utils.dataloader.os.listdir', return_value=[])
    def test_raises_if_dataset_folder_empty(self, mock_listdir, base_config):
        with pytest.raises(RuntimeError, match="Dataset folder '.+' is empty"):
            data_load(base_config)

    def test_raises_if_version_missing(self, base_config):
        base_config['dataset']['version'] = 'nonexistent'
        with pytest.raises(ValueError, match="Unknown dataset version 'nonexistent'"):
            data_load(base_config)

    def test_transform_subset_applies_transform(self):
        dummy_tensor = torch.ones(3, 32, 32)
        dummy_subset = [(dummy_tensor, 1)]

        transform = partial(add_one)

        wrapped = TransformSubset(dummy_subset, transform=transform)
        x, y = wrapped[0]

        assert torch.allclose(x, torch.full((3, 32, 32), 2.0))
        assert y == 1

    def test_transform_subset_without_transform(self):
        dummy_tensor = torch.ones(3, 32, 32)
        dummy_subset = [(dummy_tensor, 1)]
        wrapped = TransformSubset(dummy_subset, transform=None)
        x, y = wrapped[0]
        assert torch.equal(x, dummy_tensor)
        assert y == 1
