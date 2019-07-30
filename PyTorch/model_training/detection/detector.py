import albumentations as albu
import numpy as np
import torch

# from model_training.detection.retinanet_v2 import RetinaNet
from model_training.detection.retinanet import RetinaNet
from .ssd import SSD
from .detector_postprocessing import DetectorPostProcessing


class Detector(object):
    def __init__(self, config):
        super(Detector, self).__init__()
        self._init_model(config)

        self.img_size = config['img_size']
        self.multiclass_suppression = config.get('multiclass_suppression', True)
        self.transform = self._get_transform()
        self.post_processing = DetectorPostProcessing(config)

    def __call__(self, img):
        with torch.no_grad():
            x = self._process_image(img)
            if torch.cuda.is_available():
                x = x.cuda()
            loc, conf, priors = self.net(x)
        return self.post_processing(loc, conf, priors, img.shape, multiclass_suppression=self.multiclass_suppression)

    def _process_image(self, img):
        return torch.from_numpy(np.transpose(self.transform(img), (2, 0, 1))).unsqueeze(0)

    def _init_model(self, config):
        if config['name'] == 'ssd':
            self.net = SSD(config)
        else:
            self.net = RetinaNet(config, pretrained=config.get('pretrained', False))
        model_dict = torch.load(config['filepath'], map_location=None if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(model_dict['model'])
        self.net = self.net.eval()
        if torch.cuda.is_available():
            self.net = self.net.cuda()

    def _get_transform(self):
        pipeline = albu.Compose([
            albu.Resize(self.img_size, self.img_size),
            albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def process(a):
            r = pipeline(image=a)
            return r['image']

        return process
