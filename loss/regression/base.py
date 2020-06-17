from torch import nn
import torch

class L1Loss(nn.Module):
    """l1 loss

    loss = (y_pred - labels) ^ 2

    arguments:

    y_pred: shape (batch, n_lable)
    labels: shape (batch, n_lable)
    """

    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, y_pred, labels):
        return torch.abs(y_pred - labels)

class L2Loss(nn.Module):
    """l2 loss

    loss = (y_pred - labels) ^ 2

    arguments:

    y_pred: shape (batch, n_lable)
    labels: shape (batch, n_lable)
    """
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, y_pred, labels):
        return torch.pow(y_pred - labels, 2)


if __name__ == "__main__":
    a = torch.ones(size=[3, 2, 2])
    b = torch.rand_like(a)

    print(a)
    print(b)

    l1_loss = L1Loss()
    l2_loss = L2Loss()

    print(l1_loss(a, b))
    print(l2_loss(a, b))

