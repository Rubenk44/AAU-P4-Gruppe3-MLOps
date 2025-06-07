import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import deepspeed
import torch.distributed as dist
import wandb

from modelstructure import ImageNet
from utils.utils import load_config, pick_optimizer, pick_scheduler, model_export, begin_wandb
from utils.dataloader import data_load


def train(local_rank, config):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    if local_rank == 0:
        begin_wandb()  # only initialize wandb once

    model = ImageNet()
    optimizer = pick_optimizer(model, config)
    scheduler = pick_scheduler(optimizer, config)

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config["deepspeed"],
    )

    train_loader, val_loader = data_load(config)

    for epoch in range(config['train']['epochs']):
        model_engine.train()
        running_loss = 0.0

        for step, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(local_rank).to(next(model_engine.module.parameters()).dtype)
            labels = labels.to(local_rank)

            outputs = model_engine(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            running_loss += loss.item()

        if local_rank == 0:
            avg_train_loss = running_loss / len(train_loader)
            print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

            # Optional validation
            model_engine.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(local_rank).to(next(model_engine.module.parameters()).dtype)
                    labels = labels.to(local_rank)

                    outputs = model_engine(inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    val_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = 100.0 * correct / total
            avg_val_loss = val_loss / len(val_loader)

            wandb.log({
                "val_loss": avg_val_loss,
                "val_accuracy": acc,
            })
            print(f"[Epoch {epoch+1}] Val Acc: {acc:.2f}%, Val Loss: {avg_val_loss:.4f}")

    if local_rank == 0:
        model_export(model_engine.module, local_rank, config)

    dist.destroy_process_group()


def main():
    config = load_config("config.yaml")
    local_rank = int(os.environ["LOCAL_RANK"])
    train(local_rank, config)


if __name__ == "__main__":
    main()