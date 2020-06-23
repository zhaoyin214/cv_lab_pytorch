from torch import nn
from torchvision import models
from typing import Text


class BackBone(object):

    def __init__(self, backbone: nn.Module) -> None:
        self._backbone = backbone

    @property
    def output_layer_name(self, net: nn.Module):
        return list(self._backbone.name_children())[-1][0]

    @property
    def net(self) -> nn.Module:
        return self._backbone

    def modify_output_layer(self, out_planes) -> None:
        last_layer_name, last_layer = list(self._backbone.named_children())[-1]
        in_planes = last_layer.in_features

        if isinstance(last_layer, nn.Linear):
            setattr(
                self._backbone,
                last_layer_name,
                nn.Linear(
                    in_features=in_planes,
                    out_features=out_planes,
                    bias=False
                )
            )


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
        assert name in self._backbone_list, \
            "error: {} is not available".format(name)

        return self._create_backbone(name)

    def _create_backbone(self, name: Text) -> BackBone:
        backbone = BackBone(
            getattr(models, name)(pretrained=True)
        )

        return backbone


if __name__ == "__main__":

    backbone_factory = BackBoneFactory()
    for net in BackBoneFactory._backbone_list:
        backbone = backbone_factory(net)
        print(backbone.net)

    backbone = backbone_factory("resnet18")
    backbone.modify_output_layer(68)
    print(backbone.net)
