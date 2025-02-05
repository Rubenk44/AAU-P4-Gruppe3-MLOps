import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from modelstructure import ImageNet
import datetime
    
def train(dataloader, net, criterion, optimizer, epochs=10):
    print("Training")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0 
    print("Finished Training")
    return net

def model_export(torch_model):
    datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_filename = f"Model_{datetime}.onnx"
    torch_input = torch.randn(1, 1, 32, 32)
    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
    onnx_program.save(onnx_filename)
    print(f"model saved as: {"lol"}")


def main():
    transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(
        root='Lecture 1/data', train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )

    # Initialize the network, loss function, and optimizer
    net = ImageNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # Train the network
    model = train(dataloader, net, criterion, optimizer, epochs=10)


if __name__ == "__main__":
    main()