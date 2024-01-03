### cifar-10 test #######
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 96, 7, 2, 2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(3, 2, 0),

        #     nn.Conv2d(96, 256, 5, 1, 2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(3, 2, 0),

        #     nn.Conv2d(256, 384, 3, 1, 1),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(384, 384, 3, 1, 1),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(384, 256, 3, 1, 1),
        #     nn.ReLU(inplace=True)
        # )

        self.cnn0 = nn.Conv2d(3, 96, 7, 2, 2) 

        self.relu = nn.ReLU(inplace=True)

        self.pool0 = nn.MaxPool2d(3, 2, 0)

        self.cnn1 = nn.Conv2d(96, 256, 5, 1, 2)

        self.pool1 = nn.MaxPool2d(3, 2, 0)

        self.cnn2 = nn.Conv2d(256, 384, 3, 1, 1)

        self.cnn3 = nn.Conv2d(384, 384, 3, 1, 1)

        self.cnn4 = nn.Conv2d(384, 256, 3, 1, 1)

        self.fc = nn.Sequential(
            # nn.Conv2d(256*3*3, 1024, 1, 1, 0),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),
            # nn.Conv2d(1024, 512, 1, 1, 0),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # nn.Conv2d(512, 10, 1, 1, 0)
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.cnn0(x)
        x = self.relu(x)
        x = self.pool0(x)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = self.cnn4(x)
        x = self.relu(x)
        # print(x.norm(p=1))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        # x = x.reshape(x.size(0), -1)

        return x