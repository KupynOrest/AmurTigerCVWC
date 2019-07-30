import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """ IoU loss
    """

    def __init__(self):
        super(IoULoss, self).__init__()

    @staticmethod
    def iou_metric(y_pred, y_true):
        _EPSILON = 1e-6
        op_sum = lambda x: x.sum(2).sum(2)
        loss = (op_sum(y_true * y_pred) + _EPSILON) / (
                op_sum(y_true ** 2) + op_sum(y_pred ** 2) - op_sum(y_true * y_pred) + _EPSILON)

        loss = torch.mean(loss)
        return loss

    def forward(self, y_pred, y_true):
        """ Compute IoU loss
        Args:
            y_pred (torch.Tensor): predicted values
            y_true (torch.Tensor): target values
        """
        return 1 - self.iou_metric(torch.sigmoid(y_pred), y_true)


class JointLoss(nn.Module):
    """ Joint IoU & L1 loss
    """

    def __init__(self):
        super(JointLoss, self).__init__()
        self.regression_loss = nn.L1Loss()
        self.keypoint_loss = IoULoss()

    @staticmethod
    def iou_metric(y_true, y_pred):
        _EPSILON = 1e-6
        op_sum = lambda x: x.sum(2).sum(2)
        loss = (op_sum(y_true * y_pred) + _EPSILON) / (
                op_sum(y_true ** 2) + op_sum(y_pred ** 2) - op_sum(y_true * y_pred) + _EPSILON)

        loss = torch.mean(loss)
        return loss

    def forward(self, y_pred, y_true):
        """ Compute Joint loss
        Args:
            y_pred (torch.Tensor): predicted values
            y_true (torch.Tensor): target values
        """
        k_pred, em_pred = y_pred
        k_true, em_true = y_true
        return self.keypoint_loss.forward(k_pred, k_true), self.regression_loss(torch.tanh(em_pred), em_true.flatten())

