# Pytorch单机多卡标准测试   
## TREE
* 一、主机环境   
* 二、主机环境测试
* 三、Docker环境测试  
## 一、主机环境  
* CPU  
```
cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c 
20  Genuine Intel(R) CPU @ 2.40GHz
```
* Memory   
```
free -h
              total        used        free      shared  buff/cache   available
Mem:            31G        652M         19G         77M         11G         30G
Swap:          975M        192M        783M
或者cat /proc/meminfo
```
* GPU   
```
nvidia-smi
Wed May  8 21:38:46 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.183      Driver Version: 384.183      CUDA Version: 9.0      |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:06:00.0 Off |                    0 |
| N/A   37C    P0    41W / 300W |     10MiB / 32502MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  On   | 00000000:07:00.0 Off |                    0 |
| N/A   39C    P0    43W / 300W |     10MiB / 32502MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  On   | 00000000:0A:00.0 Off |                    0 |
| N/A   39C    P0    44W / 300W |     10MiB / 32502MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  On   | 00000000:0B:00.0 Off |                    0 |
| N/A   37C    P0    45W / 300W |     10MiB / 32502MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   4  Tesla V100-SXM2...  On   | 00000000:85:00.0 Off |                    0 |
| N/A   37C    P0    42W / 300W |     10MiB / 32502MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   5  Tesla V100-SXM2...  On   | 00000000:86:00.0 Off |                    0 |
| N/A   38C    P0    43W / 300W |     10MiB / 32502MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   6  Tesla V100-SXM2...  On   | 00000000:89:00.0 Off |                    0 |
| N/A   38C    P0    43W / 300W |     10MiB / 32502MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   7  Tesla V100-SXM2...  On   | 00000000:8A:00.0 Off |                    0 |
| N/A   37C    P0    42W / 300W |     10MiB / 32502MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
* OS   
``` 
head -n 1 /etc/issue
Ubuntu 16.04.5 LTS \n \l
```
* Kernal   
``` 
uname -a
Linux ubuntu 4.4.0-131-generic #157-Ubuntu SMP Thu Jul 12 15:51:36 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux
```
* [CUDA 10.0.x](https://github.com/fusimeng/ParallelComputing/blob/master/notes/cudainstall.md)   
```   
cat /usr/local/cuda/version.txt
CUDA Version 10.0.130

nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
```
* [cuDNN 7.5.x](https://github.com/fusimeng/ParallelComputing/blob/master/notes/cudainstall.md)   
``` 
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 5
#define CUDNN_PATCHLEVEL 0
--
#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
#include "driver_types.h"
```
* [Docker](https://github.com/fusimeng/ParallelComputing/blob/master/notes/docker.md)
* [Nvidia-Docker](https://github.com/fusimeng/ParallelComputing/blob/master/notes/nvdocker.md)   
## 二、主机环境测试
### 1.主机环境准备
#### （1）.安装Anaconda
参考链接：[🔗](https://github.com/fusimeng/ai_tools)    
#### （2）. 使用Anaconda，创建所需的环境   
```shell
conda create --name pytorch python=3.6
source activate pytorch
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision tensorboardx
```
```
pip list 
Package      Version 
------------ --------
certifi      2019.3.9
cffi         1.12.3  
mkl-fft      1.0.12  
mkl-random   1.0.2   
numpy        1.16.3  
olefile      0.46    
Pillow       6.0.0   
pip          19.1    
protobuf     3.7.1   
pycparser    2.19    
setuptools   41.0.1  
six          1.12.0  
tensorboardX 1.6     
torch        1.0.1   
torchvision  0.2.1   
wheel        0.33.1
```
### 2.数据准备
下载[cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)数据集，放在data目录下。   
**cifar10介绍**    
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

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
### 3.代码准备     
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
--smmg.py # 测试主程序
--smmg_dist.py # 测试主程序
--until.py # 显式库
```
 
### 4.测试
#### (1) 用法-1   
```shell
python smmg.py   
smmg.py 使用torch.nn.DataParallel方法进行数据分布式。 
```  
**参数说明：**   
```
--lr,               default=0.001, type=float, help='learning rate'
--epoch,            default=10, type=int, help='number of epochs tp train for'
--trainBatchSize,   default=1000, type=int, help='training batch size'
--testBatchSize,    default=1000, type=int, help='testing batch size'
--cuda,             default=torch.cuda.is_available(), type=bool, help='whether cuda is in use'
--log,              default="../output/smmg.pkl", type=str, help='storage logs/models'
--num_workers,      default=4, type=int, help='number of workers to load data'
--resume,           default=None, type=str, help='resume from checkpoint,such as ../output/'
--net,              default='wideresnet', type=str, help='use net '
--gpunum,           default='2', type=int, help='number of gpu , such as 2 '
--parallel,         default='dataparallel', help='way of Parallel,dataparallel or distributed'
```
#### (2)用法-2
smmg_dist.py 使用torch.nn.parallel.DistributedDataParallel方法进行数据分布式。    
```
python smmg_dist.py --gpu_device 0 1 2 3 --batch_size 768  
```
**参数说明：**  
```
--lr              default=0.1, help=''
--resume          default=None, help=''
--batch_size      type=int, default=768, help=''
--num_workers     type=int, default=4, help=''
--gpu_devices     type=int, nargs='+', default=None, help=""

--gpu             default=None, type=int, help='GPU id to use.'
--dist-url        default='tcp://127.0.0.1:3456', type=str, help=''
--dist-backend    default='gloo', type=str, help=''
--rank            default=0, type=int, help=''
--world_size      default=1, type=int, help=''
--distributed     action='store_true', help=''
```
### 5. pytorch指定显卡的几种方式   
1.直接终端中设定：   
```
CUDA_VISIBLE_DEVICES=1 python my_script.py
```
2.python代码中设定：   
```
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```
3.使用函数 set_device
```
import torch
torch.cuda.set_device(id)
```
## 三、Docker环境测试
### 1.Docker环境准备
镜像：fusimeng/ai-pytorch:16.04-10.0-3.5-1.1.0   
### 2.数据准备
同上
### 3.代码准备
同上
### 4.测试
  
-----------------------------------------------------------------------   