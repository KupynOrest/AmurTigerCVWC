from __future__ import division
import math
from math import sqrt as sqrt
from itertools import product as product
import torch
import numpy as np

_DEFAULT_SIZE = 300
_START_LAYER = 3
_NUM_BOXES = 6


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, img_size, feature_maps):
        super(PriorBox, self).__init__()
        self.image_size = img_size
        # number of priors for feature map location (either 4 or 6)
        self.scale = self.image_size / _DEFAULT_SIZE
        self.sizes = (np.array([int(x * self.scale) for x in [21, 45, 99, 153, 207, 261, 315]]) * 1.).astype('int16')
        # self.sizes = (np.array([int(x * self.scale) for x in [26, 50, 76, 100, 118, 183, 250]]) * 1.).astype('int16')
        # array([26.26923432, 50.59560579, 76.11845245, 100.08700925,
        #        118.46470771, 183.46998877])
        self.variance = [0.1, 0.2]
        self.feature_maps = feature_maps
        self.steps = self._get_steps()
        self.aspect_ratios = [[2, 3]] * _NUM_BOXES
        self.num_priors = len(self.aspect_ratios)
        self.clip = True

    def _get_steps(self):
        steps = [0] * _NUM_BOXES
        steps[_NUM_BOXES - 1] = self.image_size
        for i in range(_NUM_BOXES - 1):
            steps[_NUM_BOXES - i - 2] = pow(2, math.ceil(math.log(self.image_size) / math.log(2)) - i - 1)
        return steps

    def _get_fm_size(self):
        func = lambda x: math.ceil(float(self.image_size) / pow(2., x + _START_LAYER)) if x <= _START_LAYER + 1 else \
            math.ceil(float(self.image_size) / pow(2., (x - 1) + _START_LAYER)) - 2
        return [func(i) for i in range(_NUM_BOXES)]

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                # cx = (j + 0.5) / f_k
                cx = (j + 0.5) / f
                # cy = (i + 0.5) / f_k
                cy = (i + 0.5) / f

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.sizes[k + 1]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
