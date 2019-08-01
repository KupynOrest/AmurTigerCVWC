import numpy as np
import torch

from model_training.detection.detector_postprocessing import DetectorPostProcessing
from model_training.detection.utils.box_utils import jaccard


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class ModelAdapter(object):
    """ Class used for getting training input for model, combining losses, calculating model score
    and exporting model state
    """

    def get_metrics(self, output=None, target=None):
        raise NotImplementedError()

    def get_loss(self, combined_loss):
        raise NotImplementedError()

    @staticmethod
    def get_input(data):
        raise NotImplementedError()

    @staticmethod
    def get_model_export(net):
        raise NotImplementedError()


class JointModelAdapter(ModelAdapter):
    """ Adapter for joint emotion reocgnition and keypoints detection models
    """

    def __init__(self):
        super(JointModelAdapter, self).__init__()
        self.iou = 0

    def get_metrics(self, output=None, target=None):
        """ Calculates model IoU
        Returns:
            dict: a dictionary with 'IoU'
        """
        return {
            'IoU': self.iou
        }

    def get_loss(self, combined_loss):
        """ Combines emotion and keypoints losses
        Args:
            combined_loss (tuple): pair of localization and confidence losses
        Returns:
            torch.Tensor: detection loss
            dict: dictionary with:
                - iou_loss (float): keypoints loss
                - emotion_loss (float): L1 emotion loss
        """
        loss_dict = {'iou_loss': combined_loss[0].item(),
                     'emotion_loss': combined_loss[1].item()}
        self.iou = 1 - combined_loss[0].item()
        return combined_loss[0] + combined_loss[1], loss_dict

    @staticmethod
    def get_input(data):
        """ Prepares input for model training
        Args:
            data (tuple): pair of:
                - img (torch.Tensor): list of images
                - keypoints (list): list of keypoints targets (torch.Tensor)
                - emotion (list): list of emotion targets (torch.Tensor)
        Returns:
            torch.Tensor: tensor of images
            tuple: tuple of 2D facial keypoints and emotions(2x1) tensor targets (torch.Tensor)
        """
        img = data['img']
        inputs = img
        targets = data['keypoints']
        emotion = data['emotion'].cuda().float()
        inputs, targets = inputs.cuda().float(), targets.cuda().float()
        return inputs, (targets, emotion)

    @staticmethod
    def get_model_export(net):
        """ Returns model weights for export
        Args:
            net (torch.Module): model
        Returns:
            dict; a dictionary containing a whole state of the model
        """
        return net.module.state_dict()


class KeypointsModelAdapter(ModelAdapter):
    def __init__(self):
        super(KeypointsModelAdapter, self).__init__()
        self.iou = 0

    def get_metrics(self, output=None, target=None):
        """ Calculates model IoU
        Args:
            output (torch.Tensor): model output
            target (torch.Tensor): target
        Returns:
            dict; dictionary with 'IoU'
        """
        return {
            'IoU': self.iou
        }

    def get_loss(self, combined_loss):
        """ Gets segmentation loss
        Args:
            combined_loss (torch.Tensor): keypoints loss
        Returns:
            torch.Tensor: loss
            dict: dictionary with 'loss'
        """
        loss_dict = {'loss': combined_loss.item()}
        self.iou = 1 - combined_loss.item()
        return combined_loss, loss_dict

    @staticmethod
    def get_input(data):
        """ Prepares input for model training
        Args:
            data (dict): dictionary with:
                - image (torch.Tensor): input images
                - keypoints (torch.Tensor): target
        Returns:
            torch.Tensor: model inputs
            torch.Tensor: model targets
        """
        img = data['img']
        inputs = img
        targets = data['keypoints']
        inputs, targets = inputs.cuda().float(), targets.cuda().float()
        return inputs, targets

    @staticmethod
    def get_model_export(net):
        """ Returns model weights for export
        Args:
            net (torch.Module): model
        Returns:
            dict; a dictionary containing a whole state of the model
        """
        return net.module.state_dict()


# TODO ( AGarg ): Check details with Orest on what each method call is reporting and getting called from.
class HeadSegmentationModelAdapter(ModelAdapter):
    def __init__(self):
        super(HeadSegmentationModelAdapter, self).__init__()
        self.iou = 0

    @staticmethod
    def _iou_binary(preds, labels, EMPTY=1., per_image=False):
        """
        IoU for foreground class
        binary: 1 foreground, 0 background
        """
        preds = ((preds.float().view(-1, 1, 1, 1)).data > 0).float()
        labels = labels.view(-1, 1, 1, 1)
        if not per_image:
            preds, labels = (preds,), (labels,)
        ious = []
        ious_th = []
        for pred, label in zip(preds, labels):
            intersection = ((label == 1) & (pred == 1)).sum()
            union = ((label == 1) | (pred == 1)).sum()
            if not union:
                iou = EMPTY
            else:
                iou = intersection.item() / union.item()
            thresholds = torch.arange(0.5, 1, 0.05)
            iou_th = []
            for thresh in thresholds:
                iou_th.append(iou > thresh)

            ious_th.append(np.mean(iou_th))
            ious.append(iou)

        iou = mean(ious)
        iou_th = mean(ious_th)
        return 100 * iou, 100 * iou_th

    def get_metrics(self, output=None, target=None):
        iou, thresholded_iou = self._iou_binary(output, target)
        return{
            'IoU': iou,
            'Thresholded_IoU': thresholded_iou
        }

    def get_loss(self, combined_loss):
        loss_dict = {'loss': combined_loss.item()}
        return combined_loss, loss_dict

    @staticmethod
    def get_input(data):
        inputs = torch.from_numpy(data['img'])
        targets = torch.from_numpy(data['mask'])
        inputs, targets = inputs.type(torch.FloatTensor).cuda(), targets.type(torch.FloatTensor).cuda()
        return inputs, targets

    @staticmethod
    def get_model_export(net):
        return net.module.state_dict()


class ClassificationModelAdapter(ModelAdapter):
    def __init__(self):
        super(ClassificationModelAdapter, self).__init__()

    def get_metrics(self, output=None, target=None):
        """ Calculates model Acc
        Args:
            output (torch.Tensor): model output
            target (torch.Tensor): target
        Returns:
            dict; dictionary with 'Acc'
        """
        def get_acc(output=None, target=None):
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = target.data.cpu().numpy()
            return np.mean((output == label).astype(int))

        return {
            'Acc': get_acc(output, target)
        }

    def get_loss(self, combined_loss):
        """ Gets segmentation loss
        Args:
            combined_loss (torch.Tensor): loss
        Returns:
            torch.Tensor: loss
            dict: dictionary with 'loss'
        """
        loss_dict = {'loss': combined_loss.item()}
        return combined_loss, loss_dict

    @staticmethod
    def get_input(data):
        """ Prepares input for model training
        Args:
            data (dict): dictionary with:
                - image (torch.Tensor): input images
                - category (torch.Tensor): target
        Returns:
            torch.Tensor: model inputs
            torch.Tensor: model targets
        """
        img = data['img']
        inputs = img
        targets = data['category']
        inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    @staticmethod
    def get_model_export(net):
        """ Returns model weights for export
        Args:
            net (torch.Module): model
        Returns:
            dict; a dictionary containing a whole state of the model
        """
        return net.module.state_dict()


class DetectionModelAdapter(ModelAdapter):
    """ Adapter for object detection models
    """

    def __init__(self, config):
        """
        Args:
            config (dict): dictionary with:
                - img_size (int): image size
                - num_classes (int): number of classes in dataset
        """
        super(DetectionModelAdapter, self).__init__()
        self.config = config
        self.post_processing = DetectorPostProcessing(self.config)
        self.img_size = self.config['img_size']
        self.num_classes = self.config['num_classes']
        self.iou_thr = 0.5

    @staticmethod
    def get_input(data):
        """ Prepares input for model training
        Args:
            data (tuple): pair of:
                - images (torch.Tensor): list of images
                - targets (list): list of targets (torch.Tensor)
        Returns:
            torch.Tensor: tensor of images
            list: list of object detection targets (torch.Tensor)
        """
        images, targets = data
        images = images.cuda()
        targets = [ann.cuda() for ann in targets]
        return images, targets

    @staticmethod
    def get_loss(combined_loss):
        """ Combines localization and classification losses
        Args:
            combined_loss (tuple): pair of localization and confidence losses
        Returns:
            torch.Tensor: detection loss
            dict: dictionary with:
                - localization (float): localization loss
                - confidence (float): classification loss
        """
        loss_dict = {'localization': combined_loss[0].item(),
                     'confidence': combined_loss[1].item()}
        return combined_loss[0] + combined_loss[1], loss_dict

    @staticmethod
    def get_model_export(net):
        """ Returns model weights for export
        Args:
            net (torch.Module): model
        Returns:
            dict; a dictionary containing a whole state of the model
        """
        return net.module.state_dict()

    def get_metrics(self, output=None, target=None):
        """ Calculates model mAP
        Args:
            output (tuple): A tuple containing localization and classification predictions and prior boxes from model
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            target (list): list of targets (torch.Tensor)
        Returns:
            dict: a dictionary with 'mAP'
        """
        return {
            'mAP': self._get_map_metric(output, target)
        }

    def _get_map_metric(self, output=None, target=None):
        """ Calculates model mAP
        Args:
            output (tuple): A tuple containing localization and classification predictions and prior boxes from model
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            target (list): list of targets (torch.Tensor)
        Returns:
            float: mAP metric
        """
        def calculate_ap(curr_output, curr_target):
            ap = 0
            unique_classes = 0
            for curr_class in range(self.num_classes):
                preds, targets = curr_output[curr_output[:, 4] == curr_class], curr_target[
                    curr_target[:, 4] == curr_class]
                if targets.size(0) == 0:
                    continue
                else:
                    unique_classes += 1
                if preds.size(0) == 0:
                    continue
                _, idx = torch.sort(preds[:, -1])
                preds = preds[idx]

                jac = jaccard(preds[:, :4], targets[:, :4]) >= self.iou_thr
                jac_new = torch.zeros(jac.shape, dtype=torch.int8)
                first_ones = [(e, i) for (i, e) in enumerate(list(jac.argmax(0).numpy()))]
                for idx in first_ones:
                    jac_new[idx] = jac[idx]

                results = [0] + list(jac_new.cumsum(0).sum(1).numpy())
                all_images = jac_new.size(1)
                correct_preds = 0
                curr_ap = 0

                for i in range(1, len(results)):
                    correct_preds += results[i] > results[i - 1]
                    curr_ap += (results[i] - results[i - 1]) / all_images * correct_preds / i
                ap += curr_ap
            return ap / unique_classes

        loc, conf, priors = output
        priors = priors[0]
        img_size = self.img_size, self.img_size
        num = len(loc)
        map_metric = 0

        for i in range(num):
            bboxes, labels, scores = self.post_processing(loc[i].unsqueeze(0), conf[i].unsqueeze(0), priors, img_size)
            bboxes, labels, scores = torch.Tensor(bboxes), torch.Tensor(labels), torch.Tensor(scores)
            bboxes, labels, scores = bboxes.view(-1, 4), labels.view(-1, 1), scores.view(-1, 1)

            labels -= 1
            bboxes /= self.img_size
            bboxes[:, 2] += bboxes[:, 0]
            bboxes[:, 3] += bboxes[:, 1]
            curr_output, curr_target = torch.cat((bboxes, labels, scores), dim=1), target[i].cpu()

            map_metric += calculate_ap(curr_output, curr_target)
        return map_metric / num


def get_model_adapter(config):
    """ Creates model adapter from config
    Args:
        config (dict): dictionary with:
            - name (str): model name
            and other adapter configs for specified name
    Returns:
        ModelAdapter: adapter for specified model
    """
    if config['task'] == 'joint':
        return JointModelAdapter()
    elif config['task'] == 'keypoints':
        return KeypointsModelAdapter()
    elif config['task'] == 'headsegmentation':
        return HeadSegmentationModelAdapter()
    elif config['task'] == 'detect':
        return DetectionModelAdapter(config['model'])
    return ClassificationModelAdapter()
