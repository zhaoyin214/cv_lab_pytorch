from torchvision import models
from typing import Text

class BackBoneFactory(object):

    _backbone_list = [
        "alexnet",
        "densenet121", "densenet161", "densenet169", "densenet201",
        "googlenet",
        "mnasnet0_5", "mnasnet1_0",
        "mobilenet_v2",
        "resnet101", "resnet152", "resnet18", "resnet34", "resnet50", "resnext101_32x8d", "resnext50_32x4d", "wide_resnet101_2", "wide_resnet50_2",
        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
        "squeezenet1_0", "squeezenet1_1",
        "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
    ]

    _no_checkpoint = [
        "mnasnet0_5", "mnasnet1_3"
    ]

    _not_support = [
        "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",
    ]

    def __call__(self, name: Text):
        assert name in self._backbone_list, "error: {} is not available".format(name)
        return getattr(models, name)(pretrained=True)

if __name__ == "__main__":

    backbone = BackBoneFactory()
    for net in BackBoneFactory._backbone_list:
        print(backbone(net))
