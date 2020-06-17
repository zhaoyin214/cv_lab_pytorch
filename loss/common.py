from torch import nn
from abc import ABCMeta, abstractmethod
from typing import Callable

class ILoss(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def set_loss_func(self, func: Callable):
        """set loss function"""
        pass
