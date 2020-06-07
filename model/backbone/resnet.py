from .basic_convs import conv1x1, conv3x3

from torch import nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes: int, planes: int,
        stride: int=1, groups :int=1, base_width :int=64, dilation :int=1,
        downsample :nn.Module=None, norm_layer :nn.Module=nn.BatchNorm2d
    ):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(
            in_planes=in_planes, out_planes=planes, stride=stride
        )
        self.bn1 = norm_layer(num_features=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes)
        self.bn2 = norm_layer(num_features=planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self, in_planes: int, planes: int,
        stride: int=1, groups :int=1, base_width :int=64, dilation :int=1,
        downsample :nn.Module=None, norm_layer :nn.Module=nn.BatchNorm2d
    ):
        width = int(planes * (base_width / 64)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_planes=in_planes, out_planes=width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(
            in_planes=width, out_planes=width,
            stride=stride, padding=dilation, groups=groups, dilation=dilation
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out