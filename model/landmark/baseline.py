from torch import nn

from ..common import BaseModel

class BaselineNet(BaseModel):

    def __init__(self, backbone: nn.Module):
        super(BaselineNet, self).__init__()
        self.backbone = backbone