# -*- coding: utf-8 -*-
'''Train CIFAR10 with PyTorch.'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

# 1.参数解析
parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epoch', default=10, type=int, help='number of epochs tp train for')
parser.add_argument('--trainBatchSize', default=1000, type=int, help='training batch size')
parser.add_argument('--testBatchSize', default=1000, type=int, help='testing batch size')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
parser.add_argument('--log', default="../output/smmg.pkl", type=str, help='storage logs/models')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers to load data')   
parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint,such as ../output/')
parser.add_argument('--net', default='wideresnet', type=str, help='use net ')
parser.add_argument('--gpunum', default='2', type=int, help='number of gpu , such as 2 ')
parser.add_argument('--parallel', default='dataparallel', help='way of Parallel,dataparallel or distributed')
args = parser.parse_args()

epoch = int(args.epoch)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# 2. 数据集准备
print('====>>>> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.trainBatchSize, shuffle=True, num_workers=args.num_workers)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.testBatchSize, shuffle=False, num_workers=args.num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 3. 网络准备
print('====>>>> Building model..')
if args.net == "wideresnet":
    net = WideResNet(depth=28, num_classes=10)
else:
    net = WideResNet(depth=28, num_classes=10)

device = 'cuda' if args.cuda else 'cpu'
if device == 'cuda':
    gpulist = list(range(args.gpunum))
    if args.parallel == "dataparallel":
        print("====>>>>running on:",gpulist)
        net = torch.nn.DataParallel(net, device_ids = gpulist).cuda() # make parallel
        # net = net.to(device)
        cudnn.benchmark = True
    elif args.parallel == "distributed":
        net = torch.nn.DataParallel(net) # make parallel
    else:
        net = torch.nn.DataParallel(net) # make parallel
        # net = torch.nn.DataParallel(ResNet(ResidualBlock, [2, 2, 2, 2]),device_ids=gpulist).cuda()
    
if  args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.exists(args.resume), 'Error: no checkpoint file found!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# 4.网络训练
def train(epoch):
    print('\n====>>>> Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

# 5.模型测试
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('====>>>> Model Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        torch.save(state, args.log)
        best_acc = acc


if __name__=="__main__":
    for epoch in range(start_epoch, start_epoch + args.epoch):
        trainloss = train(epoch)
        test(epoch)
    
