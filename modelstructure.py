'''
import torch.nn as nn
import torch.nn.functional as F


class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''

import torch.nn as nn
import torch.nn.functional as F


class ImageNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Conv layer 1
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)  # Conv layer 2

        # Fully connected layers (we define fc1 dynamically later)
        self.fc1 = None
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Dynamically calculate flatten size
        if self.fc1 is None:
            flattened_size = x.view(x.size(0), -1).shape[1]
            self.fc1 = nn.Linear(flattened_size, 120).to(x.device)

        x = x.view(x.size(0), -1)  # Flatten before FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
