from typing import Callable

from .common import ILoss

class FaceAlignmentLoss(ILoss):
    def __init__(self, loss_func: Callable=None) -> None:
        super(FaceAlignmentLoss, self).__init__()

        self._loss_func = loss_func


    def forward(self, y_pred, labels):
        loss = self._loss_func(y_pred, labels)
        return loss

    def set_loss_func(self, loss_func: Callable):
        self._loss_func = loss_func