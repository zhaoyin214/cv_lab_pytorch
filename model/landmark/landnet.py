from torch import nn
import math

from ..base import batchnorm, conv1x1, conv3x3_block, maxpool2x2, relu
from ..common import IModel

class LandmarkNet(IModel):
    """this is a baseline model referring to
       the intel regression net of 5 landmarks
    """

    def __init__(
        self,
        num_landmarks: int=68,
        input_res: int=256,
        activation: nn.Module=relu
    ) -> None:
        super(LandmarkNet, self).__init__()
        self.input_res = input_res

        self.bn_input = batchnorm(3)

        res = input_res
        in_planes = 3
        planes = 16
        self.landnet = []
        while not res % 2:
            self.landnet.append(self._make_block(in_planes, planes))
            in_planes = planes
            planes *= 2
            res //= 2
        self.landnet = nn.Sequential(*self.landnet)

        bottleneck_planes = 1024
        bottleneck_max_res = 5
        self.pool = []
        if res > bottleneck_max_res:
            self.pool.append(nn.AdaptiveAvgPool2d(output_size=bottleneck_max_res))
            res = bottleneck_max_res
        self.pool += [
            nn.Conv2d(planes, planes, kernel_size=res, groups=planes, padding=0),
            batchnorm(planes),
            relu(),
            conv1x1(planes, bottleneck_planes),
            batchnorm(bottleneck_planes),
            relu()
        ]
        self.pool = nn.Sequential(*self.pool)

        fc_planes = 256
        self.fc_loc = nn.Sequential(
            conv1x1(bottleneck_planes, fc_planes),
            relu(),
            conv1x1(fc_planes, num_landmarks),
            nn.Sigmoid()
        )

    def _make_block(self, in_planes: int, planes: int):
        return nn.Sequential(
            conv3x3_block(in_planes, planes),
            conv3x3_block(planes, planes),
            maxpool2x2(),
        )

    def forward(self, x):
        x = self.bn_input(x)
        x = self.landnet(x)
        x = self.pool(x)
        x = self.fc_loc(x)
        return x

    def get_input_res(self):
        return self.input_res

    def set_dropout_ratio(self, ratio: float):
        pass

if __name__ == "__main__":
    landnet = LandmarkNet(input_res=448)
    print(landnet)