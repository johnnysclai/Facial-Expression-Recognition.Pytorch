'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--bs', default=128, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 44
total_epoch = 250

path = os.path.join(opt.dataset + '_' + opt.model)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = FER2013(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=1)

# Model
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model  == 'Resnet18':
    net = ResNet18()

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
else:
    print('==> Building model..')

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    pbar = tqdm(trainloader, ncols=0)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        description = 'Loss: %.3f | Acc: %.3f (%d/%d)'\
                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        pbar.set_description(desc=description)

    Train_acc = 100.*correct/total


def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    predicted_labels, labels = None, None
    pbar = tqdm(PublicTestloader, ncols=0)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(outputs_avg, targets)
            PublicTest_loss += loss.item()
            _, predicted = torch.max(outputs_avg.data, 1)
            if batch_idx == 0:
                predicted_labels = predicted.detach().cpu().numpy().reshape((-1, 1))
                labels = targets.detach().cpu().numpy().reshape((-1, 1))
            else:
                predicted_labels = np.vstack((predicted_labels, predicted.detach().cpu().numpy().reshape((-1, 1))))
                labels = np.vstack((labels, targets.detach().cpu().numpy().reshape((-1, 1))))
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
            description = 'Loss: %.3f | Acc: %.3f (%d/%d)'\
                           % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total)
            pbar.set_description(desc=description)
        cmatrix = confusion_matrix(labels, predicted_labels, labels=np.arange(7))
        y_true_sum = cmatrix.sum(axis=1).clip(min=1e-12)
        cmatrix_normalized = cmatrix / y_true_sum
        mean_acc = cmatrix_normalized.diagonal().mean()
        message = "Mean accuracy (fer2013_PublicTest): {}".format(mean_acc)
        print(message)
    PublicTest_acc = 100.*correct/total


def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    predicted_labels, labels = None, None
    pbar = tqdm(PrivateTestloader, ncols=0)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            bs, ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(outputs_avg, targets)
            PrivateTest_loss += loss.item()
            _, predicted = torch.max(outputs_avg.data, 1)
            if batch_idx == 0:
                predicted_labels = predicted.detach().cpu().numpy().reshape((-1, 1))
                labels = targets.detach().cpu().numpy().reshape((-1, 1))
            else:
                predicted_labels = np.vstack((predicted_labels, predicted.detach().cpu().numpy().reshape((-1, 1))))
                labels = np.vstack((labels, targets.detach().cpu().numpy().reshape((-1, 1))))
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            description = 'Loss: %.3f | Acc: %.3f (%d/%d)' \
                          % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total)
            pbar.set_description(desc=description)
        cmatrix = confusion_matrix(labels, predicted_labels, labels=np.arange(7))
        y_true_sum = cmatrix.sum(axis=1).clip(min=1e-12)
        cmatrix_normalized = cmatrix / y_true_sum
        mean_acc = cmatrix_normalized.diagonal().mean()
        message = "Mean accuracy (fer2013_PrivateTest): {}".format(mean_acc)
        print(message)
    PrivateTest_acc = 100.*correct/total


for epoch in range(start_epoch, total_epoch):
    train(epoch)
    PublicTest(epoch)
    PrivateTest(epoch)
