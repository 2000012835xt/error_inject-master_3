### cifar-10 test #######
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 6, 5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(6, 16, 5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )

        self.cnn0 = nn.Conv2d(3, 6, 5)

        self.cnn1 = nn.Conv2d(6, 16, 5)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.cnn0(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
