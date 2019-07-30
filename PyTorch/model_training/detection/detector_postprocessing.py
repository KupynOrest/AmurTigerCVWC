from functools import partial

import torch
import torch.nn.functional as F


def _decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    """Soft NMS implementation.
    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        max_score_index = torch.argmax(box_scores[:, 4])
        cur_box_prob = torch.tensor(box_scores[max_score_index, :])
        picked_box_scores.append(cur_box_prob)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.tensor([])


def _nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep, 0
    x1, y1, x2, y2 = map(lambda x: boxes[:, x], range(4))
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        xx1, yy1, xx2, yy2 = map(lambda x: torch.index_select(x, 0, idx), (x1, y1, x2, y2))
        # store element-wise max with next highest score
        xx1, yy1 = map(lambda x: torch.clamp(x[0], min=x[1][i]), zip([xx1, yy1], [x1, y1]))
        xx2, yy2 = map(lambda x: torch.clamp(x[0], max=x[1][i]), zip([xx2, yy2], [x2, y2]))
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w, h = map(partial(torch.clamp, min=0.0), (w, h))
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        iou = inter / union  # store result in iou
        idx = idx[iou.le(overlap)]
    return keep, count


def get_area(left_top, right_bottom) -> torch.Tensor:
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = get_area(overlap_left_top, overlap_right_bottom)
    area0 = get_area(boxes0[..., :2], boxes0[..., 2:])
    area1 = get_area(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def _nms_mean(boxes, scores, overlap=0.5, top_k=200):
    picked_indexes = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:top_k]
    while len(indexes) > 0:
        current = indexes[0]
        picked_indexes.append([current.item()])
        if 0 < top_k == len(picked_indexes) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        picked_indexes[-1].extend(indexes[iou > overlap].detach().cpu().numpy())
        indexes = indexes[iou <= overlap]

    averaged_boxes = []

    for indexes_group in picked_indexes:
        scores_group = scores[indexes_group]
        score = scores_group[0]
        boxes_group = boxes[indexes_group]

        # weigh picked bboxes using their scores
        scores_group = scores_group / scores_group.sum()
        averaged_box = (scores_group.unsqueeze(-1) * boxes_group).sum(0)
        averaged_boxes.append(torch.cat([averaged_box, score.unsqueeze(0)]))

    res = torch.stack(averaged_boxes)
    return res


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, variance=(0.1, 0.2), bkg_label=0, top_k=200, conf_thresh=0.01, nms_thresh=0.45):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = variance

    def __call__(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = _decode(loc_data[i], prior_data, self.variance).cpu()
            # For each class, perform nms
            conf_scores = conf_preds[i].cpu()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = _nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


class DetectorPostProcessing(object):
    def __init__(self, config):
        self.detect = Detect(config['num_classes'], conf_thresh=config.get('filter_thr', 0.01))
        self.visual_thr = config["visual_thr"]

    def __call__(self, loc, conf, priors, img_shape, multiclass_suppression=True):
        bboxes, labels, scores = self.get_detections(loc, conf, priors, img_shape)
        if multiclass_suppression:
            bboxes, labels, scores = self.multiclass_suppression(bboxes, labels, scores)
        return bboxes, labels, scores

    def get_detections(self, loc, conf, priors, img_shape):
        detections = self.detect(loc, F.softmax(conf, dim=-1), priors)

        # scale each detection back up to the image
        scale = torch.Tensor([img_shape[1], img_shape[0], img_shape[1], img_shape[0]])

        bboxes, labels, scores = [], [], []
        # ii -> category id, 0 - background class
        for ii in range(1, detections.size(1)):
            j = 0
            while detections[0, ii, j, 0] >= self.visual_thr:
                score = detections[0, ii, j, 0].tolist()
                pt = (detections[0, ii, j, 1:] * scale).tolist()
                bbox = [pt[0], pt[1], pt[2] - pt[0], pt[3] - pt[1]]

                bboxes.append(bbox)
                labels.append(ii)
                scores.append(score)
                j += 1
        return bboxes, labels, scores

    def multiclass_suppression(self, bboxes, labels, scores, nms_thresh=0.5):
        bboxes, labels, scores = torch.Tensor(bboxes), torch.IntTensor(labels), torch.Tensor(scores)
        # ids, count = _nms(bboxes, scores, nms_thresh, top_k=len(bboxes))
        # to_keep = ids[:count]
        # bboxes, labels, scores = bboxes[to_keep].tolist(), labels[to_keep].tolist(), scores[to_keep].tolist()

        ids, count = _nms(bboxes, scores, nms_thresh, top_k=len(bboxes))
        to_keep = ids[:count]
        bboxes, labels, scores = bboxes[to_keep].tolist(), labels[to_keep].tolist(), scores[to_keep].tolist()

        return bboxes, labels, scores
