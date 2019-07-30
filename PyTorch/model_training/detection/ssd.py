import torch
import torch.nn as nn

from .backbones import get_backbone
from .prior_box import PriorBox


class SSD(nn.Module):
    def __init__(self, config):
        super(SSD, self).__init__()
        self.num_classes = config["num_classes"]
        self.size = config["img_size"]

        # SSD network
        self.backbone, layer_dims = get_backbone(config)
        self.extras = SSD._create_extra(layer_dims[1])
        self.loc, self.conf = SSD._create_multibox(self.num_classes, layer_dims)

        self.priorbox = PriorBox(self.size)

        with torch.no_grad():
            self.priors = self.priorbox.forward()
            if torch.cuda.is_available():
                self.priors = self.priors.cuda()
        print(self.priors.size())

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        x = self.backbone['layer0'](x)
        x = self.backbone['layer1'](x)
        x = self.backbone['layer2'](x)
        sources.append(x)
        x = self.backbone['layer3'](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for layer in self.extras:
            x = layer(x)
            sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            self.priors
        )
        return output

    @staticmethod
    def _create_extra(in_channels):
        layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                nn.ReLU()
            )
        ])
        return layers

    @staticmethod
    def _create_multibox(num_classes, layer_dims):
        regression_headers = nn.ModuleList([
            nn.Conv2d(in_channels=layer_dims[0], out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=layer_dims[1], out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=1),
        ])

        classification_headers = nn.ModuleList([
            nn.Conv2d(in_channels=layer_dims[0], out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=layer_dims[1], out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=1),
        ])
        return regression_headers, classification_headers


def build_ssd(config):
    return nn.DataParallel(SSD(config))
