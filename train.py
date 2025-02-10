import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from modelstructure import ImageNet
from datetime import datetime
import json
from utils import device_conf, load_config
import wandb
import os


def get_run_number():
    wandb_dir = "wandb"
    if not os.path.exists(wandb_dir):
        return 1
    runs = [d for d in os.listdir(wandb_dir) if os.path.isdir(os.path.join(wandb_dir, d))]
    return len(runs)


def train(train_loader, val_loader, device, epochs=10):
    print("Training")

    net = ImageNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        num_batches = len(train_loader)

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % max(1, num_batches // 5) == 0:
                print(f'[{epoch + 1}, {i + 1}/{num_batches}] loss: {running_loss / (i + 1):.3f}')
        
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1) 
                correct += (predicted == labels).sum().item() 
                total += labels.size(0)

        avg_train_loss = running_loss / num_batches
        avg_val_loss = val_loss / len(val_loader)
        acc_val = correct / total * 100

        print(f'Epoch {epoch + 1} - Training Loss: {avg_train_loss:.3f}, Validation Loss: {avg_val_loss:.3f}, Accuracy: {acc_val:.2f}%')

        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "training_loss": avg_train_loss,
            "validation_loss": avg_val_loss,
            "validation_accuracy": acc_val
        })

        scheduler.step()

    print("Finished Training")
    return net


def model_export(model, config, device):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_filename = f"Model_{timestamp}.onnx"
    config_filename = f"Config_{onnx_filename}.txt"

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
        }
    )
    print(f"Model & config file saved as: {onnx_filename} & {config_filename}")


def main():
    config = load_config("config.yaml")
    torch.manual_seed(42)
    device = device_conf()

    run_number = get_run_number()
    wandb.login()
    wandb.init(project='DAKI4_testing', name=f"Job {run_number}")

    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)

    model = train(train_loader, val_loader, device, epochs=10)
    model_export(model, config, device)


if __name__ == "__main__":
    main()