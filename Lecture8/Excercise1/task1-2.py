import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
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


def filter_dataset(dataset, digits):
    idx = [i for i, (img, label) in enumerate(dataset) if label in digits]
    return Subset(dataset, idx)


def train(model, device, train_loader, optimizer, epoch, log_interval, dry_run):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                f'Train Epoch: {epoch} '
                f'[{batch_idx * len(data)}/'
                f'{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                f'Loss: {loss.item():.6f}'
            )

            if dry_run:
                break


def test(model, device, test_loader, label_name="Test"):
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
            total += len(target)

    accuracy = 100.0 * correct / total
    avg_loss = test_loss / total

    output_str = (
        f'{label_name}: Avg loss: {avg_loss:.4f}, '
        f'Accuracy: {correct}/{total} '
        f'({accuracy:.2f}%)'
    )
    print(output_str)

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

    use_accel = not args.no_accel and torch.cuda.is_available()
    device = torch.device("cuda" if use_accel else "cpu")

    begin_wandb()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        '../data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    train_0_4 = filter_dataset(train_dataset, digits=[0, 1, 2, 3, 4])
    train_5_9 = filter_dataset(train_dataset, digits=[5, 6, 7, 8, 9])
    test_0_4 = filter_dataset(test_dataset, digits=[0, 1, 2, 3, 4])
    test_5_9 = filter_dataset(test_dataset, digits=[5, 6, 7, 8, 9])

    train_loader_0_4 = torch.utils.data.DataLoader(
        train_0_4, batch_size=args.batch_size, shuffle=True
    )
    train_loader_5_9 = torch.utils.data.DataLoader(
        train_5_9, batch_size=args.batch_size, shuffle=True
    )
    test_loader_0_4 = torch.utils.data.DataLoader(
        test_0_4, batch_size=args.test_batch_size, shuffle=False
    )
    test_loader_5_9 = torch.utils.data.DataLoader(
        test_5_9, batch_size=args.test_batch_size, shuffle=False
    )

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    wandb.watch(model)

    print("\n--- Training on digits 0–4 ---")
    epochs_task1 = (args.epochs + 1) // 2
    for epoch in range(1, epochs_task1 + 1):
        train(
            model,
            device,
            train_loader_0_4,
            optimizer,
            epoch,
            args.log_interval,
            args.dry_run,
        )
        acc_0_4, loss_0_4 = test(model, device, test_loader_0_4, "Digits 0-4")
        wandb.log(
            {
                "Task 1 Epoch": epoch,
                "Accuracy 0-4": acc_0_4,
                "Loss 0-4": loss_0_4,
            }
        )
        scheduler.step()
        if args.dry_run:
            break

    print("\n--- Continue Training on digits 5–9 (without reset) ---")
    forgetting = []
    epochs_task2 = args.epochs - epochs_task1
    for epoch in range(1, epochs_task2 + 1):
        train(
            model,
            device,
            train_loader_5_9,
            optimizer,
            epoch,
            args.log_interval,
            args.dry_run,
        )
        acc_0_4, loss_0_4 = test(
            model, device, test_loader_0_4, label_name="Digits 0-4 (Old)"
        )
        acc_5_9, loss_5_9 = test(
            model, device, test_loader_5_9, label_name="Digits 5-9 (New)"
        )
        forgetting.append(acc_0_4)
        wandb.log(
            {
                "Task 2 Epoch (Forgetting)": epoch,
                "Accuracy 0-4 (Forgetting)": acc_0_4,
                "Loss 0-4": loss_0_4,
                "Accuracy 5-9 (Forgetting)": acc_5_9,
                "Loss 5-9": loss_5_9,
            }
        )
        scheduler.step()
        if args.dry_run:
            break

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    wandb.finish()


if __name__ == '__main__':
    main()
