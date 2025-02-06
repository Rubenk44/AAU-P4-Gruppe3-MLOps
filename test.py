import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import onnxruntime as ort

def main():
    torch.manual_seed(42)

    transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )


if __name__ == "__main__":
    main()