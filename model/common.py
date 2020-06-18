from torch import nn
from abc import ABCMeta, abstractmethod

class BaseModel(nn.Module, metaclass=ABCMeta):
    """the abstract model class
    """

    @abstractmethod
    def set_dropout_ratio(self, ratio: float):
        """set dropout ratio of the model"""

    @abstractmethod
    def get_input_res(self):
        """returns input resolution"""

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    tensor=m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)
