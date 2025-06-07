import os
import torch
import torchvision.transforms as transforms
import pytest
from unittest import mock
from unittest.mock import patch, MagicMock

from utils.dataloader import add_transform, TransformSubset, data_load


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


class TestDataLoad:
    """Test suite for data_load function"""

    @patch('utils.dataloader.AugmentedCIFAR10', autospec=True)
    @patch('torchvision.datasets.CIFAR10', autospec=True)
    @patch('utils.dataloader.AugmentedCIFAR10.__init__', return_value=None)
    @patch('torchvision.datasets.CIFAR10.__init__', return_value=None)
    @patch('torchvision.datasets.CIFAR10._check_integrity', return_value=True)
    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('utils.dataloader.random_split')
    @patch('utils.dataloader.TransformSubset')
    @patch('utils.dataloader.DataLoader')
    def test_data_load_raises_if_folder_empty(
        self,
        mock_dataloader,
        mock_transform_subset,
        mock_split,
        mock_mkdir,
        mock_exists,
        mock_listdir,
        mock_cifar_init,
        mock_check_integrity,
        mock_augmented_cifar_init,
        mock_cifar_class,
        mock_augmented_cifar_class,
    ):
        """Test data_load raises if dataset folder is empty and not downloaded"""
        mock_exists.return_value = True
        mock_listdir.return_value = []

        config = {
            'dataset': {
                'version': 'original',
                'paths': {
                    'original': {'dvc_target': './data.dvc', 'local_path': './data'}
                },
            },
            'train': {'train_test_split': 0.8, 'batch_size': 32, 'num_workers': 2},
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

        with patch('builtins.print'), patch('utils.dataloader.add_transform'):
            with pytest.raises(RuntimeError, match="Dataset folder './data' is empty"):
                data_load(config)

    @patch('utils.dataloader.AugmentedCIFAR10', autospec=True)
    @patch('torchvision.datasets.CIFAR10', autospec=True)
    @patch('utils.dataloader.AugmentedCIFAR10.__init__', return_value=None)
    @patch('torchvision.datasets.CIFAR10.__init__', return_value=None)
    @patch('torchvision.datasets.CIFAR10._check_integrity', return_value=True)
    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('utils.dataloader.random_split')
    @patch('utils.dataloader.TransformSubset')
    @patch('utils.dataloader.DataLoader')
    def test_data_load_folder_not_exists(
        self,
        mock_dataloader,
        mock_transform_subset,
        mock_split,
        mock_mkdir,
        mock_exists,
        mock_listdir,
        mock_cifar_init,
        mock_check_integrity,
        mock_augmented_cifar_init,
        mock_cifar_class,
        mock_augmented_cifar_class,
    ):
        """Test data_load when data folder doesn't exist"""

        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_dataloader.side_effect = [mock_train_loader, mock_val_loader]

        config = {
            'dataset': {
                'version': 'original',
                'paths': {
                    'original': {'dvc_target': './data.dvc', 'local_path': './data'}
                },
            },
            'train': {'train_test_split': 0.8, 'batch_size': 32, 'num_workers': 2},
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

        mock_exists.side_effect = [False, True]
        mock_listdir.return_value = ['some_file']

        mock_tensor = torch.randn(3, 32, 32)

        mock_cifar_instance = mock_cifar_class.return_value
        mock_cifar_instance.__len__.return_value = 1000
        mock_cifar_instance.__getitem__.return_value = (mock_tensor, 0)
        mock_cifar_instance.data = [mock_tensor] * 1000

        mock_augmented_instance = mock_augmented_cifar_class.return_value
        mock_augmented_instance.__len__.return_value = 1000
        mock_augmented_instance.__getitem__.return_value = (mock_tensor, 0)
        mock_augmented_instance.data = [mock_tensor] * 1000

        mock_split.return_value = (MagicMock(), MagicMock())
        mock_transform_subset.side_effect = [MagicMock(), MagicMock()]

        with patch('utils.dataloader.add_transform') as mock_add_transform:
            mock_add_transform.return_value = (
                transforms.ToTensor(),
                transforms.ToTensor(),
            )
            with patch('builtins.print'):
                train_loader, val_loader = data_load(config)

        mock_mkdir.assert_called_once_with('./data')
        assert mock_dataloader.call_count == 2


@patch('utils.dataloader.AugmentedCIFAR10', autospec=True)
@patch('torchvision.datasets.CIFAR10', autospec=True)
@patch('utils.dataloader.AugmentedCIFAR10.__init__', return_value=None)
@patch('torchvision.datasets.CIFAR10.__init__', return_value=None)
@patch('torchvision.datasets.CIFAR10._check_integrity', return_value=True)
@patch('os.listdir')
@patch('os.path.exists')
@patch('utils.dataloader.random_split')
@patch('utils.dataloader.TransformSubset')
@patch('utils.dataloader.DataLoader')
def test_data_load_different_split_ratio(
    mock_dataloader,
    mock_transform_subset,
    mock_split,
    mock_exists,
    mock_listdir,
    mock_cifar_init,
    mock_check_integrity,
    mock_augmented_cifar_init,
    mock_cifar_class,
    mock_augmented_cifar_class,
):
    """Test data_load with different train/test split ratio"""
    mock_train_loader = MagicMock()
    mock_train_loader.batch_size = 16
    mock_val_loader = MagicMock()
    mock_val_loader.batch_size = 16
    mock_dataloader.side_effect = [mock_train_loader, mock_val_loader]

    config = {
        'dataset': {
            'version': 'original',
            'paths': {'original': {'dvc_target': './data.dvc', 'local_path': './data'}},
        },
        'train': {'train_test_split': 0.7, 'batch_size': 16, 'num_workers': 4},
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

    mock_tensor = torch.randn(3, 32, 32)
    mock_exists.return_value = True
    mock_listdir.return_value = ['data']

    mock_cifar_instance = mock_cifar_class.return_value
    mock_cifar_instance.__len__.return_value = 1000
    mock_cifar_instance.__getitem__.return_value = (mock_tensor, 0)
    mock_cifar_instance.data = [mock_tensor] * 1000

    mock_augmented_instance = mock_augmented_cifar_class.return_value
    mock_augmented_instance.__len__.return_value = 1000
    mock_augmented_instance.__getitem__.return_value = (mock_tensor, 0)
    mock_augmented_instance.data = [mock_tensor] * 1000

    mock_train_subset = MagicMock()
    mock_train_subset.__len__.return_value = 700
    mock_val_subset = MagicMock()
    mock_val_subset.__len__.return_value = 300

    mock_split.return_value = (mock_train_subset, mock_val_subset)
    mock_transform_subset.side_effect = [MagicMock(), MagicMock()]

    with patch('utils.dataloader.add_transform') as mock_add_transform:
        mock_add_transform.return_value = (transforms.ToTensor(), transforms.ToTensor())
        with patch('builtins.print'):
            train_loader, val_loader = data_load(config)

    mock_split.assert_called_once_with(mock_cifar_instance, [700, 300])
    assert train_loader.batch_size == 16
    assert val_loader.batch_size == 16
