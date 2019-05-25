# -*- coding: utf-8 -*-
# @Author  : Miaoshuyu
# @Email   : miaohsuyu319@163.com
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x) + F.relu(-x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


dummy_input = torch.rand(13, 1, 28, 28)

model = Net1()
with SummaryWriter(comment='Net1') as w:
    w.add_graph(model, (dummy_input,))
'''
首先我们定义一个神经网络取名为Net1。
然后将其添加到tensorboard可是可视化中。
with SummaryWriter(comment='Net1')as w:         
w.add_graph(model, (dummy_input,))      

我们重点关注最后两句话，其中使用了python的上下文管理，with 语句，可以避免因w.close未写造成的问题。推荐使用此方式。       
因为这是一个神经网络架构，所以使用 w.add_graph(model, (dummy_input,))，
其中第一个参数为需要保存的模型，第二个参数为输入值，元祖类型。打开tensorvboard控制台，
'''