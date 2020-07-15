from abc import ABCMeta, abstractmethod
from torch import nn
from typing import Dict, List
import torch

class BaseFaceAlignLoss(nn.Module, metaclass=ABCMeta):
    def __init__(self, norm_ref: Dict=None) -> None:
        super(BaseFaceAlignLoss, self).__init__()
        self.set_norm_ref(norm_ref)

    def forward(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        loss = self._loss(preds, labels)
        loss /= self._norm_distance(labels)
        return torch.sum(loss) / loss.shape[0]

    @abstractmethod
    def _loss(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        pass

    def _norm_distance(self, labels: torch.Tensor) -> torch.Tensor:
        norm_dist = torch.zeros(labels.shape[0])
        for pt_pair in zip(self._left_pts, self._right_pts):
            norm_dist += self._norm_dist_op(labels, pt_pair)
        return norm_dist / len(self._left_pts)

    def set_norm_ref(self, norm_ref: Dict=None):
        self._left_pts = norm_ref["left"]
        self._right_pts = norm_ref["right"]

    @abstractmethod
    def _norm_dist_op(
        self, labels: torch.Tensor, pt_pair: List
    ) -> torch.Tensor:
        pass

class FaceAlignL1Loss(BaseFaceAlignLoss):

    def _norm_dist_op(
        self, labels: torch.Tensor, pt_pair: List
    ) -> torch.Tensor:
        return torch.abs(
            labels[:, 2 * pt_pair[1]] - labels[:, 2 * pt_pair[0]]
        ) + torch.abs(
            labels[:, 2 * pt_pair[1] + 1] - labels[:, 2 * pt_pair[0] + 1]
        )

    def _loss(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return torch.norm(input=labels - preds, p=1, dim=1) / labels.shape[0]

class FaceAlignL2Loss(BaseFaceAlignLoss):
    def _norm_dist_op(
        self, labels: torch.Tensor, pt_pair: List
    ) -> torch.Tensor:
        return torch.square(
            labels[:, 2 * pt_pair[1]] - labels[:, 2 * pt_pair[0]]
        ) + torch.square(
            labels[:, 2 * pt_pair[1] + 1] - labels[:, 2 * pt_pair[0] + 1]
        )

    def _loss(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return torch.norm(
            input=labels - preds, p=2, dim=1
        ).pow_(2) / labels.shape[0]


class FaceAlignSmoothL1Loss(BaseFaceAlignLoss):
    def set_norm(self):
        pass

class FaceAlignWingLoss(BaseFaceAlignLoss):
    """
    Wing Loss for Robust Facial Landmark Localisation with
    Convolutional Neural Networks
    https://arxiv.org/abs/1711.06753
    """
    def wing_core(self, abs_x, w, eps):
        return w * torch.log(1. + abs_x / eps)

    def _norm_dist_op(
        self, labels: torch.Tensor, pt_pair: List
    ) -> torch.Tensor:
        return torch.abs(
            labels[:, 2 * pt_pair[1]] - labels[:, 2 * pt_pair[0]]
        ) + torch.abs(
            labels[:, 2 * pt_pair[1] + 1] - labels[:, 2 * pt_pair[0] + 1]
        )



if __name__ == "__main__":

    labels = torch.zeros(5, 20)
    labels[:, 10 : 20] = 1
    preds = torch.randn(5, 20)
    norm_ref = {
        "left": [2, 3],
        "right": [7, 8]
    }
    l1_loss = FaceAlignL1Loss(norm_ref)
    print(l1_loss(preds, labels))

    l2_loss = FaceAlignL2Loss(norm_ref)
    print(l2_loss(preds, labels))
