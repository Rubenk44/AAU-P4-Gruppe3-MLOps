import torch
import torch.nn as nn
from modelstructure import ImageNet


# Test: Kontroller, at forward-pass fungerer
def test_forward_pass():
    model = ImageNet()
    model.eval()

    input_data = torch.randn(4, 3, 32, 32)

    output = model(input_data)

    assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"


def test_training_step():
    model = ImageNet()
    model.train()

    # Dummy input og labels
    input_data = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))

    # Loss-funktion og optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Forward-pass
    output = model(input_data)
    loss = criterion(output, labels)

    # Backward-pass og optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Test: Loss skal være en positiv værdi
    assert loss.item() > 0, "Loss skal være positiv efter en training step"
