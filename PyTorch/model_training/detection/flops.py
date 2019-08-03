import cv2
import torch
from thop import profile, clever_format

from model_training.detection.ssd import build_ssd
from model_training.detection.retinanet import build_retinanet
from model_training.detection.config import get_config

cv2.setNumThreads(0)


def _get_model(config):
    model_config = config['model']
    if model_config['name'] == 'ssd':
        model = build_ssd(model_config)
    elif model_config['name'] == 'retina_net':
        model = build_retinanet(model_config, parallel=False)
    else:
        raise ValueError("Model [%s] not recognized." % model_config['name'])
    return model


if __name__ == '__main__':
    config = get_config("config/train.yaml")
    model = _get_model(config).to('cuda:0')
    img_size = 256
    input = torch.randn(1, 3, img_size, img_size).to('cuda:0')
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('Flops - {}'.format(flops))
    print('Params - {}'.format(params))
