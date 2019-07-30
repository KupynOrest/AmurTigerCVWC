import logging
from collections import defaultdict

import numpy as np
from tensorboardX import SummaryWriter

import os.path as osp


class MetricCounter:
    """ Class which is used to store metrics and write it to tensorboard
    """

    def __init__(self, exp_config):
        """
            Args:
                exp_config (dict): dictionary with:
                    - name (str): experiment name
                    - folder (str): folder where to save results
        """
        self.writer = SummaryWriter(osp.join(exp_config['folder'], exp_config['name']))
        logging.basicConfig(filename=osp.join(exp_config['folder'], exp_config['name'], 'model.log'), level=logging.DEBUG)
        self.metrics = defaultdict(list)
        self.best_metric = 0
        self.window_size = 100

    def add_losses(self, loss_dict):
        """ Adds given losses to cache
        Args:
            loss_dict (dict): dictionary with losses
        """
        raise NotImplementedError()

    def loss_message(self):
        """ Creates loss message from cache
        Returns:
            str: loss message
        """
        raise NotImplementedError()

    def get_metric(self):
        """ Calculates metric from cached list of metrics
        Returns:
            float: metric value
        """
        raise NotImplementedError()

    def clear(self):
        """ Clears metrics cache
        """
        self.metrics = defaultdict(list)

    def write_to_tensorboard(self, epoch_num, validation=False):
        """ Writes cached metrics to tensorboard
        Args:
            epoch_num (int): epoch number
            validation (bool): True if it is validation metrics, else False
        """
        scalar_prefix = 'Validation' if validation else 'Train'
        for k in [key for key in self.metrics.keys() if key != 'default']:
            self.writer.add_scalar('{}_{}'.format(scalar_prefix, k), np.mean(self.metrics[k]), epoch_num)

    def get_loss(self):
        """ Calculated loss from cached losses
        Returns:
            float: loss
        """
        return np.mean(self.metrics['Loss'])

    def add_metrics(self, metric_dict):
        """ Adds metrics to local cache
        Args:
            metric_dict (dict): dictionary with metrics
        """
        for metric_name in metric_dict:
            self.metrics[metric_name].append(metric_dict[metric_name])

    def update_best_model(self):
        """ Checks, which current model is the best
        Returns:
            bool: True if model has the best metric, else False
        """
        cur_metric = self.get_metric()
        if self.best_metric < cur_metric:
            self.best_metric = cur_metric
            return True
        return False



class DetectionMetricCounter(MetricCounter):
    def __init__(self, exp_config):
        MetricCounter.__init__(self, exp_config)

    def add_losses(self, loss_dict):
        self.metrics['Loss'].append(loss_dict['localization'] + loss_dict['confidence'])
        self.metrics['LocationLoss'].append(loss_dict['localization'])
        self.metrics['ConfidenceLoss'].append(loss_dict['confidence'])

    def loss_message(self):
        metrics = ((k, np.mean(self.metrics[k][-self.window_size:])) for k in ('Loss',))
        return '; '.join(map(lambda x: '{}={:.4f}'.format(x[0], x[1]), metrics))

    def get_metric(self):
        return np.mean(self.metrics['mAP'])


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):

        score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_metric_counter(config):
    return DetectionMetricCounter(config['experiment'])
