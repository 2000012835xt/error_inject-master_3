#!/usr/bin/env python3

"""
Define the Task class for Cifar10
"""

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from task import TaskBase
from utils import KLLossSoft


class Cifar100Task(TaskBase):

    def __init__(self, data_root='/home/zzd/zzd_data/PycharmProjects/data'):
        self.data_root = data_root
        self.criterion = nn.CrossEntropyLoss()
        self.kd_criterion = KLLossSoft()

    def get_train_transform(self):
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def get_test_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def get_train_dataloader(self, batch_size, num_workers=2):
        transform = self.get_train_transform()
        dataset = torchvision.datasets.CIFAR100(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

    def get_test_dataloader(self, batch_size, num_workers=20):
        transform = self.get_train_transform()
        dataset = torchvision.datasets.CIFAR100(
            root=self.data_root,
            train=False,
            download=True,
            transform=transform,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

    def get_criterion(self, use_kd=False):
        return self.kd_criterion if use_kd else self.criterion
