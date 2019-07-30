import torch
import torch.nn as nn

from model_training.detection.utils.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """ SSD Multibox Loss.
    Basically, Multibox loss combines classification loss and localization regression loss.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, localization_loss, confidence_loss, variance=(0.1, 0.2), mining="hard",
                 overlap_thresh=0.5, bkg_label=0, neg_pos=3):
        """
        Args:
            localization_loss (torch.nn.Module): localization loss
            confidence_loss (torch.nn.Module): classification loss
            variance (tuple): variances
            mining (str): type of mining technique:
                - hard: is used to suppress the presence of a large number of negative prediction
                - none: do not use mining techniques
            overlap_thresh (float): overlap threshold used to match prior boxes with ground-truth values
            bkg_label (int): background class id
            neg_pos (int): negative-positive ration for hard negative mining
        """
        super(MultiBoxLoss, self).__init__()
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.neg_pos_ratio = neg_pos
        self.variance = variance

        self.confidence_loss = confidence_loss
        self.localization_loss = localization_loss

        self.mining = self.get_mining(mining)

    def forward(self, predictions, targets):
        """Compute classification loss and smooth l1 loss.
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            targets (batch_size,num_objs,5): Ground truth boxes and labels for a batch (last idx is the label).
        """
        loc_p, conf_p, priors = predictions
        num_classes = conf_p.size(2)

        # match priors (default boxes) and ground truth boxes
        loc_t, conf_t = self.encode_target(loc_p, targets, priors)

        # Compute max conf across batch for hard negative mining
        with torch.no_grad():
            mask = self.mining(conf_p, conf_t, self.neg_pos_ratio)

        # Localization Loss
        pos_mask = conf_t > 0
        loc_p = loc_p[pos_mask, :].view(-1, 4)
        loc_t = loc_t[pos_mask, :].view(-1, 4)
        loss_l = self.localization_loss(loc_p, loc_t)

        # Confidence Loss
        conf_p = conf_p[mask, :].view(-1, num_classes)
        conf_t = conf_t[mask]
        loss_c = self.confidence_loss(conf_p, conf_t)

        N = pos_mask.long().sum(1, keepdim=True).data.sum()  # conf_t.size(0)
        return loss_l / N, loss_c / N

    def encode_target(self, loc_data, targets, priors):
        """Produces localization targets
        Args:
            loc_data (batch_size,num_priors,num_classes): predicted border boxes
            targets (batch_size,num_objs,5): ground truth boxes and labels (last idx is the label)
            priors (num_priors,4): prior boxes
        """
        with torch.no_grad():
            batch_size = loc_data.size(0)
            num_priors = (priors.size(0))

            loc_t = torch.Tensor(batch_size, num_priors, 4)
            conf_t = torch.LongTensor(batch_size, num_priors)

            for idx in range(batch_size):
                truths = targets[idx][:, :-1].data
                labels = targets[idx][:, -1].data
                defaults = priors.data
                match(self.threshold, truths, defaults, self.variance, labels,
                      loc_t, conf_t, idx)

            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        return loc_t, conf_t

    @staticmethod
    def get_mining(method):
        if method == "none":
            mining = none_mining
        elif method == "hard":
            mining = hard_negative_mining
        else:
            raise ValueError("Mining method [%s] not recognized." % method)
        return mining


def hard_negative_mining(conf_p, conf_t, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        conf_p (N, num_priors): predicted labels.
        conf_t (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    num_classes = conf_p.size(2)
    batch_conf = conf_p.view(-1, num_classes)
    loss = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
    loss = loss.view(conf_t.size()[0], conf_t.size()[1])
    pos_mask = conf_t > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -float("inf")
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


def none_mining(conf_p, conf_t, neg_pos_ratio):
    return torch.ones_like(conf_t, dtype=torch.uint8)
