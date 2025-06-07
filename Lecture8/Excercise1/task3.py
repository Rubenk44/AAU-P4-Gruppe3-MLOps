import ssl
import random
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import wandb
from Lecture8.Mnistmodel import Net
from avalanche.benchmarks import nc_benchmark
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    forgetting_metrics,
)
from avalanche.logging import InteractiveLogger

ssl._create_default_https_context = ssl._create_unverified_context


def main():
    parser = argparse.ArgumentParser(
        description='Avalanche MNIST Task 3 with Replay + wandb'
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--replay-mem-size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--wandb-project', type=str, default="MLOps")
    parser.add_argument('--wandb-run-name', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    mnist_train = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    scenario = nc_benchmark(
        mnist_train,
        mnist_test,
        n_experiences=2,
        task_labels=False,
        seed=args.seed,
        shuffle=False,
    )

    model = Net().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)

    replay_plugin = ReplayPlugin(mem_size=args.replay_mem_size)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )

    cl_strategy = Naive(
        model,
        optimizer,
        F.nll_loss,
        train_mb_size=args.batch_size,
        train_epochs=args.epochs,
        eval_mb_size=1000,
        device=device,
        plugins=[replay_plugin],
        evaluator=eval_plugin,
    )

    print("Starting training with Avalanche experience replay")
    for experience in scenario.train_stream:
        print(f"Start training on experience {experience.current_experience}")
        cl_strategy.train(experience)
        print("Training completed.")

        print("Evaluating on test stream")
        results = cl_strategy.eval(scenario.test_stream)
        print(results)

        wandb.log(
            {
                "Accuracy_0-4": results.get('Top1_Acc_Exp0_Stream0', None),
                "Loss_0-4": results.get('Loss_Exp0_Stream0', None),
                "Accuracy_5-9": results.get('Top1_Acc_Exp1_Stream0', None),
                "Loss_5-9": results.get('Loss_Exp1_Stream0', None),
                "Experience": experience.current_experience,
                "Epoch": cl_strategy.clock.train_exp_epochs,
            }
        )

    wandb.finish()


if __name__ == '__main__':
    main()
