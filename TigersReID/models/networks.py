import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d
from models.net_sphere import SphereNet

class PlainNet(nn.Module):
    def __init__(self):
        super(PlainNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, 2, 1)  # =>B*64*256*256
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1)  # =>B*128*128*128
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1)  # =>B*128*128*128
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_5 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1)  # =>B*256*64*64
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*64*64
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*64*64
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*32*32
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1)  # =>B*512*16*16
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_3 = nn.PReLU(512)

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
        return x

class EmbeddingNet(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(EmbeddingNet, self).__init__()
        print("Backbone: ", backbone)
        self.convnet, shape, self.add_bn = self._get_convnet(backbone, pretrained)
        if self.add_bn:
            self.bn = nn.BatchNorm2d(shape[0])
        self.fc = nn.Sequential(nn.Linear(shape[0] * shape[1], 512),
                                nn.BatchNorm1d(num_features=512)
                                )
        self.dropout = nn.Dropout(0.4)

        if pretrained:
            for param in self.convnet.parameters():
                param.requires_grad = False

    def load_weights(self, path, from_net):
        if from_net == 'sphere_net':
            pretrained_dict = torch.load(path)['model']
            pretrained_dict = {k[21:]: v for k, v in pretrained_dict.items() if 'module.fc.' not in k}
            self.load_state_dict(pretrained_dict)
        else:
            self.load_state_dict(torch.load(path)['model'])

    def unfreeze(self):
        for param in self.convnet.parameters():
            param.requires_grad = True

    def forward(self, x):
        output = self.convnet(x)
        if self.add_bn:
            output = self.bn(output)
        output = self.dropout(output)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def _get_convnet(self, backbone, pretrained):
        if backbone == 'resnet34':
            convnet = nn.Sequential(*list(models.resnet34(pretrained=pretrained).children())[:-1])
            shape = (512, 4 * 5)
            add_bn = True
        elif backbone == 'resnet50':
            convnet = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-1])
            shape = (2048, 4 * 5)
            add_bn = True
        elif backbone == 'resnet101':
            convnet = nn.Sequential(*list(models.resnet101(pretrained=pretrained).children())[:-1])
            shape = (2048, 4 * 5)
            add_bn = True
        elif backbone == 'resnet152':
            convnet = nn.Sequential(*list(models.resnet152(pretrained=pretrained).children())[:-1])
            shape = (2048, 4 * 5)
            add_bn = True
        elif backbone == 'densenet121':
            convnet = nn.Sequential(models.densenet121(pretrained=pretrained).features,
                                    nn.ReLU(inplace=True),
                                    nn.AvgPool2d(7, stride=1)
                                    )
            shape = (1024, 4 * 5)
            add_bn = False
        elif backbone == 'densenet201':
            convnet = nn.Sequential(models.densenet201(pretrained=pretrained).features,
                                    nn.ReLU(inplace=True),
                                    nn.AvgPool2d(7, stride=1)
                                    )
            shape = (1920, 4 * 5)
            add_bn = False
        elif backbone == 'se_resnext_50':
            pretrain = 'imagenet' if pretrained else None
            convnet = nn.Sequential(*list(se_resnext50_32x4d(num_classes=1000, pretrained=pretrain).children())[:-1])
            shape = (2048, 4*5)
            add_bn = True
        elif backbone == 'se_resnext_101':
            pretrain = 'imagenet' if pretrained else None
            convnet = nn.Sequential(*list(se_resnext101_32x4d(num_classes=1000, pretrained=pretrain).children())[:-1])
            shape = (2048, 4 * 5)
            add_bn = True
        elif backbone == 'sphere_net':
            convnet = PlainNet()
            shape = (512, 16*16)
            add_bn = False
        else:
            raise ValueError("Backbone [%s] not recognized." % backbone)
        return convnet, shape, add_bn

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, input):
        x1, x2, x3 = input
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

def get_net(model_config, load_weights):
    model_name = model_config['name']
    if model_name == 'sphere_net':
        model = SphereNet(EmbeddingNet(model_config['backbone'], model_config['pretrained']), model_config['loss'], classnum=model_config['num_classes'])
    elif model_name == 'triplet_net':
        model = EmbeddingNet(model_config['backbone'], model_config['pretrained'])
        if model_config['selection'] == 'random':
            model = TripletNet(model)
    else:
        raise ValueError("Network [%s] not recognized." % model_name)
    if load_weights['load']:
        model.load_weights(load_weights['path'], load_weights['from'])
    return nn.DataParallel(model)


