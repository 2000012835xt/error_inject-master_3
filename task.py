#!/usr/bin/env python3

"""
Define the base class for a Task

TODO: move to pytorch lightning
"""

from abc import ABC, abstractmethod


class TaskBase(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def get_train_dataloader(self, batch_size):
        raise NotImplementedError

    @abstractmethod
    def get_test_dataloader(self, batch_size):
        raise NotImplementedError

    @abstractmethod
    def get_criterion(self):
        raise NotImplementedError

    @abstractmethod
    def get_train_transform(self):
        raise NotImplementedError

    @abstractmethod
    def get_test_transform(self):
        raise NotImplementedError
