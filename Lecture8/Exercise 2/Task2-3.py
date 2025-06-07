import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import ssl
import wandb

# Set root path and import
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from utils.utils import begin_wandb
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from Mnistmodel import Net

ssl._create_default_https_context = ssl._create_unverified_context

def set_seed(seed):
    import random
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
            accuracy = 100. * correct / total
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAcc: {accuracy:.2f}%')
            wandb.log({
                "Train Loss": avg_loss,
                "Train Accuracy": accuracy,
                "Epoch": epoch,
                "Batch": batch_idx
            })

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
    accuracy = 100. * correct / total
    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')

    if epoch is not None:
        wandb.log({
            "Test Loss": avg_loss,
            "Test Accuracy": accuracy,
            "Epoch": epoch
        })

    return accuracy, avg_loss

def unlearn(model, device, train_loader, target_class, ascent_steps=5, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    ascent_step = 0

    for data, target in train_loader:
        mask = target == target_class
        if mask.sum() == 0:
            continue

        data, target = data[mask].to(device), target[mask].to(device)
        for _ in range(ascent_steps):
            optimizer.zero_grad()
            output = model(data)
            loss = -F.nll_loss(output, target)  # Negative for gradient ascent
            loss.backward()
            optimizer.step()
            ascent_step += 1

        if ascent_step >= ascent_steps:
            break

def evaluate_forgetting(model, device, test_loader, forgotten_digit):
    model.eval()
    correct_forgotten = 0
    total_forgotten = 0
    correct_others = 0
    total_others = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            mask_forgotten = target == forgotten_digit
            correct_forgotten += pred[mask_forgotten].eq(target[mask_forgotten].view_as(pred[mask_forgotten])).sum().item()
            total_forgotten += mask_forgotten.sum().item()

            mask_others = target != forgotten_digit
            correct_others += pred[mask_others].eq(target[mask_others].view_as(pred[mask_others])).sum().item()
            total_others += mask_others.sum().item()

    forgotten_acc = 100. * correct_forgotten / total_forgotten if total_forgotten > 0 else 0
    others_acc = 100. * correct_others / total_others if total_others > 0 else 0

    print(f"\nEvaluation After Unlearning:")
    print(f"\u2192 Accuracy on forgotten digit {forgotten_digit}: {forgotten_acc:.2f}% ({correct_forgotten}/{total_forgotten})")
    print(f"\u2192 Accuracy on remaining digits: {others_acc:.2f}% ({correct_others}/{total_others})")

    wandb.log({
        f"Accuracy on Forgotten Digit {forgotten_digit}": forgotten_acc,
        "Accuracy on Remaining Digits": others_acc
    })

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example with Targeted Unlearning')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
    parser.add_argument('--epochs', type=int, default=14, metavar='N')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M')
    parser.add_argument('--no-accel', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--unlearn-digit', type=int, default=7, help='Digit to unlearn')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    begin_wandb()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    wandb.watch(model)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, args.log_interval)
        test(model, device, test_loader, epoch)
        scheduler.step()

    if args.unlearn_digit is not None:
        print(f"\nPerforming targeted unlearning on digit {args.unlearn_digit}...")
        unlearn(model, device, train_loader, args.unlearn_digit)
        evaluate_forgetting(model, device, test_loader, args.unlearn_digit)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn_baseline.pt")

    wandb.finish()

if __name__ == '__main__':
    main()