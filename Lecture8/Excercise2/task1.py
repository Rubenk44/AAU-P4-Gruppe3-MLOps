import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import ssl
import wandb
from utils.utils import begin_wandb
from Lecture8.Mnistmodel import Net
import random

ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/'
                f'{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                f'Loss: {loss.item():.6f}\tAcc: {accuracy:.2f}%'
            )
            wandb.log(
                {
                    "Train Loss": avg_loss,
                    "Train Accuracy": accuracy,
                    "Epoch": epoch,
                    "Batch": batch_idx,
                }
            )


def test(model, device, test_loader, epoch=None):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = test_loss / total
    accuracy = 100.0 * correct / total
    print(
        f'Test set: Average loss: {avg_loss:.4f}, '
        f'Accuracy: {correct}/{total} ({accuracy:.2f}%)'
    )

    if epoch is not None:
        wandb.log({"Test Loss": avg_loss, "Test Accuracy": accuracy, "Epoch": epoch})

    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)',
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        metavar='N',
        help='input batch size for testing (default: 1000)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=14,
        metavar='N',
        help='number of epochs to train (default: 14)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1.0,
        metavar='LR',
        help='learning rate (default: 1.0)',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.7,
        metavar='M',
        help='Learning rate step gamma (default: 0.7)',
    )
    parser.add_argument('--no-accel', action='store_true', help='disables accelerator')
    parser.add_argument(
        '--dry-run', action='store_true', help='quickly check a single pass'
    )
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status',
    )
    parser.add_argument(
        '--save-model', action='store_true', help='For Saving the current Model'
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    begin_wandb()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        '../data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False
    )

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    wandb.watch(model)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, args.log_interval)
        test(model, device, test_loader, epoch)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn_baseline.pt")

    wandb.finish()


if __name__ == '__main__':
    main()
