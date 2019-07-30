import torch
import torch.nn as nn

from model_training.detection.prior_box import PriorBox
from .backbones import get_backbone


class RetinaNet(nn.Module):

    def __init__(self, config, pretrained=True):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        num_anchors = config.get('num_anchors', 6)
        num_filters_fpn = config.get('num_filters_fpn', 128)
        self.fpn = FPN(config=config, num_filters=num_filters_fpn, pretrained=pretrained)
        self.num_classes = config['num_classes']
        feature_maps = [40, 20, 10, 10, 10, 10]
        self.size = config["img_size"]
        self.priorbox = PriorBox(self.size, feature_maps=feature_maps)
        self.regression = nn.ModuleList([self._make_head(num_anchors*4, x) for x in range(len(feature_maps))])
        self.classification = nn.ModuleList([self._make_head(num_anchors*self.num_classes, x) for x in range(len(feature_maps))])

        with torch.no_grad():
            self.priors = self.priorbox.forward()
            if torch.cuda.is_available():
                self.priors = self.priors.cuda()

    @staticmethod
    def _make_head(out_planes, x):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(128))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(128, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):

        maps = self.fpn(x)
        loc = list()
        conf = list()
        assert len(maps) == len(self.regression)
        assert len(self.regression) == len(self.classification)
        for m in maps:#, regr_head, cl_head in zip(maps, self.regression, self.classification):
            loc.append(self.regression[0](m).permute(0, 2, 3, 1).contiguous())
            conf.append(self.classification[0](m).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            self.priors
        )
        return output


class FPN(nn.Module):

    def __init__(self, config, num_filters=128, pretrained=True):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        self.backbone, layer_dims = get_backbone(config)
        self.conv5 = nn.Conv2d(in_channels=160, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.lateral4 = nn.Conv2d(160, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(layer_dims[1], num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(layer_dims[0], num_filters, kernel_size=1, bias=False)

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(num_filters),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        # Bottom-up pathway, from ResNet
        enc0 = self.backbone['layer0'](x)
        enc1 = self.backbone['layer1'](enc0)  # 256
        enc2 = self.backbone['layer2'](enc1)  # 512
        enc3 = self.backbone['layer3'](enc2)  # 1024
        enc4 = self.backbone['layer4'](enc3)  # 2048
        map5 = self.conv5(enc4)
        map6 = self.conv6(torch.relu(map5))
        map7 = self.conv7(torch.relu(map6))

        # Lateral connections
        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)

        # Top-down pathway
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest"))
        map2 = self.td2(lateral2 + nn.functional.upsample(map3, scale_factor=2, mode="nearest"))
        # for i in [map2, map3, map4, map5, map6, map7]:
        #     print(i.size())
        return map2, map3, map4, map5, map6, map7


def build_retinanet(config):
    return nn.DataParallel(RetinaNet(config))
