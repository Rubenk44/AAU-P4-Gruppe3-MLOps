import torch
import torch.nn as nn
from modelstructure import ImageNet
from utils import (
    load_config,
    device_conf,
    data_load,
    model_export,
    begin_wandb,
    pick_optimizer,
    pick_scheduler,
)
import wandb


def train(train_loader, val_loader, device, config):
    print("Training")

    model = ImageNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = pick_optimizer(model, config)
    scheduler = pick_scheduler(optimizer, config)

    for epoch in range(config['train']['epochs']):
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader)

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % max(1, num_batches // 5) == 0:
                print(
                    f'[{epoch + 1}, {i + 1}/{num_batches}] '
                    f'loss: {running_loss / (i + 1):.3f}'
                )

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_train_loss = running_loss / num_batches
        avg_val_loss = val_loss / len(val_loader)
        acc_val = correct / total * 100

        print(
            f"Epoch {epoch + 1}, "
            f"Training Loss: {avg_train_loss:.3f}, "
            f"Validation Loss: {avg_val_loss:.3f}, "
            f"Accuracy: {acc_val:.2f}%"
        )

        current_lr = optimizer.param_groups[0]['lr']
        wandb.log(
            {
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "training_loss": avg_train_loss,
                "validation_loss": avg_val_loss,
                "validation_accuracy": acc_val,
            }
        )

        scheduler.step()

    print("Finished Training")
    return model


def main():
    config = load_config("config.yaml")
    torch.manual_seed(42)
    device = device_conf()

    begin_wandb()

    train_loader, val_loader = data_load(config)

    model = train(train_loader, val_loader, device, config)
    model_export(model, device, config)


if __name__ == "__main__":
    main()
