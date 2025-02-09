import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from modelstructure import ImageNet
from datetime import datetime
import yaml
import json

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


def train(train_loader, val_loader, net, criterion, optimizer, device, epochs=10):
    print("Training")
    net.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999: ############################################# gør sådan den her del ændre sig an på størrelsen af datasættet
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            print(f'Epoch {epoch + 1} Validation Loss: {val_loss / len(val_loader):.3f}')

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
    
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)

    net = ImageNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    model = train(train_loader, val_loader, net, criterion, optimizer, device, epochs=1)
    model_export(model, config, device)


if __name__ == "__main__":
    main()