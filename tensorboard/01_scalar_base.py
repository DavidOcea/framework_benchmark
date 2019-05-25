# -*- coding: utf-8 -*-
# @Author  : Miaoshuyu
# @Email   : miaohsuyu319@163.com
import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir='scalar')
for epoch in range(100):
    writer.add_scalar('scalar/test', np.random.rand(), epoch)
    writer.add_scalars('scalar/scalars_test', {'xsinx': epoch * np.sin(epoch), 'xcosx': epoch * np.cos(epoch)}, epoch)

writer.close()
"""
对上述代码进行解释，首先导入：from tensorboardX import SummaryWriter，然后定义一个SummaryWriter() 实例。

在SummaryWriter()上鼠标ctrl+b我们可以看到SummaryWriter()的参数为：
def __init__(self, log_dir=None, comment='', **kwargs): 
其中log_dir为生成的文件所放的目录，comment为文件名称。默认目录为生成runs文件夹目录。


"""