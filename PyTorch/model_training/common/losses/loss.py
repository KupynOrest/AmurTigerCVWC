import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from .keypoints_losses import IoULoss, JointLoss
from .detection_losses import MultiBoxLoss


def _get_classfication_loss(loss_name):
    """ Get classification loss by name
    Args:
        loss_name (str): loss name
    """
    if loss_name == 'bce':
        loss = nn.BCEWithLogitsLoss()
    elif loss_name == 'cross_entropy_loss':
        loss = nn.CrossEntropyLoss(size_average=True)
    else:
        raise ValueError("Classification loss [%s] not recognized." % loss_name)
    return loss


def _get_localization_loss(loss_name):
    """ Gets localization loss by name
    Args:
        loss_name (str): loss name
    """
    if loss_name == 'l1_smooth_loss':
        loss = nn.SmoothL1Loss(size_average=True)
    else:
        raise ValueError("Localization loss [%s] not recognized." % loss_name)
    return loss


class JointFocalLoss(nn.Module):
    def __init__(self):
        super(JointFocalLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.iou_loss = IoULoss()

    def forward(self, y_pred, y_true):
        focal_loss = self.focal_loss.forward(y_pred, y_true)
        iou = self.iou_loss.forward(y_pred, y_true)
        return 0.66 * iou + 0.33 * focal_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.5, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def _focal_loss(self, p, q, gamma):
        return p * torch.log(q + self.eps) * (1 - q + self.eps) ** gamma

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        f_loss = self._focal_loss(y_true, y_pred, self.gamma) + self._focal_loss(1 - y_true, 1 - y_pred, self.gamma)
        return -f_loss.mean()


class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    @staticmethod
    def _lovasz_grad(gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    @staticmethod
    def _flatten_binary_scores(scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels

    def _lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
        """
        if len(labels) == 0:
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * Variable(signs))
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
        return loss

    def forward(self, y_pred, y_true):
        """
        Binary Lovasz hinge loss
            y_pred: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
            y_true: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        """
        return self._lovasz_hinge_flat(*self._flatten_binary_scores(y_pred, y_true))


def get_loss(loss):
    """ Creates loss from config
    Args:
        loss (dict): dictionary of loss configuration:
        - name (str): loss name
        and other configs for specified loss
    """
    loss_name = loss['name']
    if loss_name == 'multibox_loss':
        loss = MultiBoxLoss(localization_loss=_get_localization_loss(loss['localization_loss']),
                            confidence_loss=_get_classfication_loss(loss['confidence_loss']),
                            mining=loss.get('mining', 'hard'))
    elif loss_name == 'l2_loss':
        loss = nn.MSELoss()
    elif loss_name == 'iou_loss':
        loss = IoULoss()
    elif loss_name == 'joint_loss':
        loss = JointLoss()
    elif loss_name == 'cross_entropy':
        loss = nn.CrossEntropyLoss()
    elif loss_name == 'bce':
        loss = nn.BCEWithLogitsLoss()
    elif loss_name == 'lovasz':
        loss = LovaszLoss()
    elif loss_name == 'focal_iou':
        loss = JointFocalLoss()
    elif loss_name == 'focal_loss':
        loss = FocalLoss()
    else:
        raise ValueError("Loss [%s] not recognized." % loss_name)
    return loss
