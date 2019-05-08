# pytorch单机多卡标准测试
参考：https://github.com/dnddnjs/pytorch-multigpu   
## 一、主机环境
### 1.环境准备
(1) [安装Anaconda](https://github.com/fusimeng/ai_tools)    
(2) 使用Anaconda，创建所需的环境   
* python3.6
* numpy
* pytorch 1.0.0
* torchvision 0.2.1
```shell
conda create --name pytorch python=3.6
source activate pytorch
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytorch torchvision tensorboardx
或者
conda install pytorch torchvision tensorboardx
```
```shell
pip list 
Package     Version 
----------- --------
certifi     2019.3.9
cffi        1.12.3  
mkl-fft     1.0.12  
mkl-random  1.0.2   
numpy       1.16.3  
olefile     0.46    
Pillow      6.0.0   
pip         19.1    
pycparser   2.19    
setuptools  41.0.1  
six         1.12.0  
torch       1.0.1   
torchvision 0.2.1   
wheel       0.33.1 
```
### 2.数据准备
下载[cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)数据集，放在data目录下。   
**cifar10介绍**    
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
### 3.代码准备
目录介绍：  
``` 
-pytorch  # pytorch标准测试代码目录 
--models  # 模型目录
----AlexNet.py
----DenseNet.py
----GoogleNet.py
----LeNet.py
----ResNet.py
----VGG.py
----WideResNet.py
--smsg.py # 测试主程序
--misc.py # 显式库
```
![](../imgs/01.png)  
### 4.测试及结果分析
**用法**：   
```shell
python smsg.py --epoch 1 --trainBatchSize 10000 --testBatchSize 10000
```
optional arguments:   
```
--lr                default=1e-3    learning rate
--epoch             default=200     number of epochs tp train for
--trainBatchSize    default=100     training batch size
--testBatchSize     default=100     test batch size
```
## 二、Docker环境
### 1.环境准备
镜像：https://cloud.docker.com/repository/docker/fusimeng/ai.pytorch    
使用镜像：fusimeng/ai.pytorch:v5   
### 2.数据准备
同上
### 3.代码准备
同上
### 4.测试及结果分析
```shell
nvidia-docker run -itd -v /root/felix/:/workspace fusimeng/ai.pytorch:v5
nvidia-docker exec -it xxx bash
```
**用法**：   
同上