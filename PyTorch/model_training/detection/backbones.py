from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.models as models
from model_training.common.models.mobilenet_v2 import MobileNetV2


def _create_vgg(pretrained=True):
    vgg19 = models.vgg19(pretrained=pretrained)
    return nn.ModuleDict({'layer0': vgg19.features[:7],
                          'layer1': vgg19.features[7:14],
                          'layer2': vgg19.features[14:23],
                          'layer3': vgg19.features[23:34]}), [256, 512]


def _create_mobilenet(pretrained=True, weights=None):
    dims = [32, 64]
    net = MobileNetV2(n_class=1000)

    if pretrained:
        # Load weights into the project directory
        state_dict = torch.load(
            weights)
        net.load_state_dict(state_dict)
    features = deepcopy(net.features)

    return nn.ModuleDict({'layer0': nn.Sequential(*deepcopy(features[0:2])),
                          'layer1': nn.Sequential(*deepcopy(features[2:4])),
                          'layer2': nn.Sequential(*deepcopy(features[4:7])),
                          'layer3': nn.Sequential(*deepcopy(features[7:11])),
                          'layer4': nn.Sequential(*deepcopy(features[11:16]))}), dims


def _create_resnet(encoder_depth=34, pretrained=True):
    dims = [512, 1024]
    if encoder_depth == 34:
        resnet = models.resnet34(pretrained=pretrained)
        dims = [128, 256]
    elif encoder_depth == 50:
        resnet = models.resnet50(pretrained=pretrained)
    elif encoder_depth == 101:
        resnet = models.resnet101(pretrained=pretrained)
    else:
        raise ValueError("Incorrect resnet backbone configuration")

    return nn.ModuleDict({'layer0': nn.Sequential(*[resnet.conv1,
                                                    resnet.bn1,
                                                    resnet.relu,
                                                    resnet.maxpool]),
                          'layer1': nn.Sequential(*[resnet.layer1]),
                          'layer2': nn.Sequential(*[resnet.layer2]),
                          'layer3': nn.Sequential(*[resnet.layer3]),
                          'layer4': nn.Sequential(*[resnet.layer4])}), dims


def _create_densenet(encoder_depth=121, pretrained=True):
    dims = [256, 512]
    if encoder_depth == 121:
        densenet = models.densenet121(pretrained=pretrained)
    elif encoder_depth == 169:
        densenet = models.densenet169(pretrained=pretrained)
    elif encoder_depth == 201:
        densenet = models.densenet201(pretrained=pretrained)
    else:
        raise ValueError("Incorrect densenet backbone configuration")

    densenet = densenet.features
    return nn.ModuleDict({'layer0': nn.Sequential(*[densenet.conv0,
                                                    densenet.norm0,
                                                    densenet.relu0,
                                                    densenet.pool0]),
                          'layer1': nn.Sequential(*[densenet.denseblock1,
                                                    densenet.transition1]),
                          'layer2': nn.Sequential(*[densenet.denseblock2,
                                                    densenet.transition2]),
                          'layer3': nn.Sequential(*[densenet.denseblock3,
                                                    densenet.transition3]),
                          'layer4': nn.Sequential(*[densenet.denseblock4,
                                                    ])}), dims


def get_backbone(config):
    if config["backbone"] == "vgg":
        return _create_vgg(config["pretrained"])
    elif config["backbone"] == "resnet":
        return _create_resnet(config["encoder_depth"], config["pretrained"])
    elif config["backbone"] == "mobilenet":
        return _create_mobilenet(config["pretrained"], weights=config['pretrain_weights'])
    elif config["backbone"] == "densenet":
        return _create_densenet(config["encoder_depth"], config["pretrained"])
    else:
        raise ValueError("Incorrect backbone configuration")
