import torch
import torch.nn as nn
from pytorchcv.models.mobilenet import fdmobilenet_wd2 as backbone
#from pytorchcv.models.efficientnet import efficientnet_b2b as backbone
#from pytorchcv.models.seresnext import seresnext101_32x4d as backbone

from model_training.detection.prior_box import PriorBox
from .backbones import get_backbone
import torchviz


class DConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        self.conv_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv_depthwise(x)
        x = self.conv_pointwise(x)
        return x  


class RetinaNet(nn.Module):

    def __init__(self, config, pretrained=True):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        num_anchors = config.get('num_anchors', 6)
        num_filters_fpn =96#config.get('num_filters_fpn', 128)
        self.fpn = FPN(config=config, num_filters=num_filters_fpn, pretrained=pretrained)
        self.num_classes = config['num_classes']
        feature_maps = [28, 14, 7, 7, 7, 7]
        self.size = config["img_size"]
        self.priorbox = PriorBox(self.size, feature_maps=feature_maps)
        self.regression = nn.ModuleList([self._make_head(num_anchors*4, x, num_filters_fpn) for x in range(len(feature_maps))])
        self.classification = nn.ModuleList([self._make_head(num_anchors*self.num_classes, x, num_filters_fpn) for x in range(len(feature_maps))])

        with torch.no_grad():
            self.priors = self.priorbox.forward()
            if torch.cuda.is_available():
                self.priors = self.priors.cuda()

    @staticmethod
    def _make_head(out_planes, x, num_filters):
        layers = []
        # for _ in range(2):
        #     layers.append(DConv2d(num_filters, num_filters))
        #     layers.append(nn.BatchNorm2d(num_filters))
        #     layers.append(nn.ReLU(True))
        # layers.append(DConv2d(num_filters, out_planes))
        layers.append(nn.Conv2d(num_filters, out_planes, kernel_size=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):

        maps = self.fpn(x)
        loc = list()
        conf = list()
        assert len(maps) == len(self.regression)
        assert len(self.regression) == len(self.classification)
        for m, regr_head, cl_head in zip(maps, self.regression, self.classification):
            loc.append(regr_head(m).permute(0, 2, 3, 1).contiguous())
            conf.append(cl_head(m).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            self.priors.unsqueeze(0)
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
        net = backbone(pretrained=True).features
        self.backbone = nn.ModuleList([
            nn.Sequential(net.init_block, net.stage1),
            net.stage2,
            net.stage3,
            net.stage4,
            # net.stage5
        ])
        self.conv5 = DConv2d(in_channels=512, out_channels=num_filters)
        self.conv6 = nn.Sequential(
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            DConv2d(in_channels=num_filters, out_channels=num_filters)
        )
        self.conv7 = nn.Sequential(
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            DConv2d(in_channels=num_filters, out_channels=num_filters)
        )

        self.lateral4 = nn.Conv2d(512, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(128, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)

        self.td1 = nn.Sequential(DConv2d(num_filters, num_filters),
                                 nn.BatchNorm2d(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(DConv2d(num_filters, num_filters),
                                 nn.BatchNorm2d(num_filters),
                                 nn.ReLU(inplace=True))
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        # Bottom-up pathway, from ResNet
        enc0 = self.backbone[0](x)
        enc1 = self.backbone[1](enc0)
        enc2 = self.backbone[2](enc1)
        enc3 = self.backbone[3](enc2)
        # enc4 = self.backbone[4](enc3)
        map5 = self.conv5(enc3)
        map6 = self.conv6(map5)
        map7 = self.conv7(map6)

        # Lateral connections
        lateral4 = self.lateral4(enc3)
        lateral3 = self.lateral3(enc2)
        lateral2 = self.lateral2(enc1)

        # Top-down pathway
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest"))
        map2 = self.td2(lateral2 + nn.functional.upsample(map3, scale_factor=2, mode="nearest"))
        # for i in [map2, map3, map4, map5, map6, map7]:
           # print(i.size())
        return map2, map3, map4, map5, map6, map7


def build_retinanet(config, parallel=True):
    model = RetinaNet(config)
    from torchviz import make_dot
    y = model(torch.randn(1, 3, config["img_size"], config["img_size"]))
    vis_graph = make_dot(y, params=dict(list(model.named_parameters())))
    vis_graph.format = 'svg'
    vis_graph.render()
    print('rendered')
    if parallel:
        model = nn.DataParallel(model)
    return model
