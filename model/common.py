from torch import nn
from abc import ABCMeta, abstractmethod

class IModel(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def set_dropout_ratio(self, ratio: float):
        """set dropout ratio of the model"""

    @abstractmethod
    def get_input_res(self):
        """returns input resolution"""