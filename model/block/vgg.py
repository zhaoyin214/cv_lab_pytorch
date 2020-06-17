from ..base import batchnorm, conv3x3, maxpool2x2, relu
from torch import nn
from typing import List

class VGGBlock(nn.Module):

    def __init__(
        self,
        cfg: List,
        batch_norm: bool=True, bias: bool=True
    ):
        super(VGGBlock, self).__init__()
        self.features = self._make_layers(cfg=cfg, batch_norm=batch_norm)

    def forward(self, x):
        return x

    def _make_layers(self, cfg: List, batch_norm: bool):

        layers = []
        in_planes = cfg[0]
        for v in cfg[1 :]:
            if v == 'M':
                layers.append(maxpool2x2())
            else:
                conv2d = conv3x3(in_planes=in_planes, out_planes=v)
                if batch_norm:
                    layers += [conv2d, batchnorm(v), relu()]
                else:
                    layers += [conv2d, relu()]
                in_planes = v

        return nn.Sequential(*layers)

