import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np

class OnlineTripletModel(object):
    def __init__(self):
        super(OnlineTripletModel, self).__init__()

    def get_input(self, data):
        input = torch.cat((data['img1'], data['img2'], data['img3']), 0)
        return input.cuda(), data['class']

    def get_acc(self, output=None, target=None):
        #TODO: Create some metric
        return 0

    def get_loss(self, mean_loss, mean_acc, output=None, target=None):
        return '{:.6f}'.format(mean_loss)

    def visualize_data(self, writer, data, outputs, niter):
        writer.add_image('Anchors', vutils.make_grid(data['img1']), niter)
        writer.add_image('SameClass', vutils.make_grid(data['img2']), niter)
        writer.add_image('SameClass2', vutils.make_grid(data['img3']), niter)
        writer.add_embedding(outputs.cpu(), label_img=torch.cat((data['img1'], data['img2'], data['img3']), 0), global_step=niter)

class TripletModel(object):
    def __init__(self):
        super(TripletModel, self).__init__()

    def get_input(self, data):
        data = (data['img'], data['positive_img'], data['negative_img'])
        return tuple(d.cuda() for d in data), None


    def get_acc(self, output=None, target=None):
        #TODO: Create some metric
        return 0

    def get_loss(self, mean_loss, mean_acc, output=None, target=None):
        return '{:.6f}'.format(mean_loss)

    def visualize_data(self, writer, data, outputs, niter):
        writer.add_image('Anchors', vutils.make_grid(data['img']), niter)
        writer.add_image('Positive', vutils.make_grid(data['positive_img']), niter)
        writer.add_image('Negative', vutils.make_grid(data['negative_img']), niter)

class SphereModel(nn.Module):
    def __init__(self):
        super(SphereModel, self).__init__()

    def get_input(self, data):
        img = data['A']
        inputs = img
        targets = data['id']
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        return (inputs, targets), targets

    def get_acc(self, output=None, target=None):
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = target.data.cpu().numpy()
#         print('output/label: ', output, label)
        return np.mean((output == label).astype(int))

    def get_loss(self, mean_loss, mean_acc, output=None, target=None):
        return '{:.3f}; accuracy={}'.format(mean_loss, mean_acc)

    def visualize_data(self, writer, data, outputs, niter):
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )
        images = inv_normalize(vutils.make_grid(data['A']))
        writer.add_image('Images', images, niter)

def get_model(model_config):
    model_name = model_config['name']
    if model_name == 'sphere_net':
        model = SphereModel()
    elif model_name == 'triplet_net':
        if model_config['selection'] == 'random':
            model = TripletModel()
        else:
            model = OnlineTripletModel()
    else:
        raise ValueError("Model [%s] not recognized." % model_name)
    return model


