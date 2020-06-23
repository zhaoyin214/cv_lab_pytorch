from torch import nn
from typing import Text

from ..common import BaseModel
from ..backbone import BackBone

class BaselineNet(BaseModel):

    def __init__(
        self, backbone: BackBone, num_landmarks: int=68, bottleneck: int=1
    ) -> None:
        super(BaselineNet, self).__init__()
        self._create_backbone(backbone, num_landmarks)

    def _create_backbone(self, backbone: BackBone, num_landmarks: int) -> None:
        backbone.modify_output_layer(num_landmarks * 2)
        self.backbone = backbone.net

    def forward(self, x):
        return self.backbone(x)

    def set_dropout_ratio(self):
        pass

    def get_input_res(self):
        return 224, 224



if __name__ == "__main__":

    from ..backbone import BackBoneFactory

    backbone = "resnet50"
    backbone = BackBoneFactory()(backbone)
    base_line = BaselineNet(backbone)
    print(base_line)