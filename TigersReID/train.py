from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from data.data_loader import CreateDataLoader
import tqdm
import os,sys,cv2,random,datetime
import yaml
import argparse
import numpy as np
from models.networks import get_net
from models.losses import get_loss
from models.models import get_model
from tensorboardX import SummaryWriter
import logging

logging.basicConfig(filename='train_resnet50_batch32_augs.log', level=logging.DEBUG)
writer = SummaryWriter('runs_resnet50_batch32_augs')
REPORT_EACH = 10
torch.backends.cudnn.bencmark = True
cv2.setNumThreads(0)

WARMUP_EPOCHS_NUM = 3

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.train_dataset = self._get_dataset(config, config['datasets']['train'])
        self.val_dataset = self._get_dataset(config, config['datasets']['validation'])
        self.best_acc = 0


    def train(self):
        self._init_params()
        for epoch in range(0, config['num_epochs']):
            if epoch == WARMUP_EPOCHS_NUM:
                self.net.module.unfreeze()
                self.optimizer = self._get_optim()
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 55, 70, 95], gamma=0.5)

            train_loss = self._run_epoch(epoch)
            val_loss, val_acc = self._validate(epoch)
            self.scheduler.step()

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save({
                    'model': self.net.state_dict()
                }, 'best_{}.h5'.format(self.config['experiment_desc']))
            torch.save({
                'model': self.net.state_dict()
            }, 'last_{}.h5'.format(self.config['experiment_desc']))
            print(('val_acc={}, val_loss={}, best_loss={}\n'.format(val_acc, val_loss, self.best_acc)))
            logging.debug("Experiment Name: %s, Epoch: %d, Train Loss: %.3f, Val Acc: %.3f, Val Loss: %.3f, Best Acc: %.3f" % (
               self.config['experiment_desc'], epoch, train_loss, val_acc, val_loss, self.best_acc))

    def _run_epoch(self, epoch):
        self.net = self.net.train()
        losses = []
        accuracy = []
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
        batches_per_epoch = len(self.train_dataset) / config['batch_size']
        tq = tqdm.tqdm(self.train_dataset.dataloader)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        for data in tq:
            inputs, targets = self.model.get_input(data)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.data.item())
            accuracy.append(self.model.get_acc(outputs, targets))
            mean_loss = np.mean(losses[-REPORT_EACH:])
            mean_acc = np.mean(accuracy[-REPORT_EACH:])
            if i % 100 == 0:
                writer.add_scalar('Train_Loss', mean_loss, i + (batches_per_epoch * epoch))
                writer.add_scalar('Train_Accuracy', mean_acc, i + (batches_per_epoch * epoch))
            tq.set_postfix(loss=self.model.get_loss(mean_loss, mean_acc, outputs, targets))
            i += 1
        tq.close()
        return np.mean(losses)

    def _validate(self, epoch):
        self.net = self.net.eval()
        losses = []
        accuracy = []
        tq = tqdm.tqdm(self.val_dataset.dataloader)
        tq.set_description('Validation')
        with torch.no_grad():
            for data in tq:
                inputs, targets = self.model.get_input(data)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                losses.append(loss.data.item())
                accuracy.append(self.model.get_acc(outputs, targets))
                tq.set_postfix(accuracy=np.mean(accuracy[-REPORT_EACH:]))
            val_loss = np.mean(losses)
            val_acc = np.mean(accuracy)
            writer.add_scalar('Validation_Loss', val_loss, epoch)
            writer.add_scalar('Validation_Accuracy', val_acc, epoch)
        return val_loss, val_acc

    def _get_dataset(self, config, filename):
        data_loader = CreateDataLoader(config, filename)
        return data_loader.load_data()

    def _get_optim(self):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.config['optimizer']['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _init_params(self):
        self.net = get_net(self.config['model'], self.config['load_weights'])
        self.net.cuda()
        self.model = get_model(self.config['model'])
        self.criterion = get_loss(self.config['model'])
        self.optimizer = self._get_optim()
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 55, 70, 95], gamma=0.5)


if __name__ == '__main__':
    with open('config/sphere_solver.yaml', 'r') as f:
        config = yaml.load(f)
    trainer = Trainer(config)
    trainer.train()


