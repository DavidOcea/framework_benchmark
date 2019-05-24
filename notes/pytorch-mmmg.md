# pytorch 多机多卡标准测试程序
## TREE
* 一、主机环境   
* 二、主机环境测试
* 三、Docker环境测试  
## 一、主机环境  
* Nodes     

|主机名|系统|IP|GPUs|
|:------:|:-----:|:-------:|:----:|
|ubuntu|Ubuntu16.04|192.168.31.150|Tesla P40 * 1|
|ff170|Ubuntu16.04|192.168.31.170|Tesla P40 * 1|
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
Fri May 24 13:31:05 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.79       Driver Version: 410.79       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P40           Off  | 00000000:03:00.0 Off |                  Off |
| N/A   41C    P0    45W / 250W |      0MiB / 24451MiB |      3%      Default |
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

  
### 3.标准测试-1（toy）
#### （1）代码结构
```
-pytorch
--mmmg_toy.py
```
#### （2）.用法   
**在单机中使用**   
```
Terminal 1
$ python mmmg_toy.py --rank 0 --world-size 2
Terminal 2
$ python mmmg_toy.py --rank 1 --world-size 2
```
[Terminal 1 的日志](pytorch-mmmg-log-1.md)   
[Terminal 2 的日志](pytorch-mmmg-log-2.md)
   
**在多机中使用**   
```
Machine 1 with ip 192.168.31.150

$ python mmmg_toy.py --rank 0 --world-size 2 --ip 192.168.31.150 --port 22000
Machine 2

$ python mmmg_toy.py --rank 1 --world-size 2 --ip 192.168.31.150 --port 22000
```
[Machine 1 的日志](pytorch-mmmg-log-3.md)   
[Machine 2 的日志](pytorch-mmmg-log-4.md)   
### 4.标准测试-2（mnist）
#### （1）代码结构
```
-pytorch
--mmmg_mnist.py
```
#### 2.用法
```
machine 1 ip 192.168.31.150 
python mmmg_mnist.py --init-method tcp://192.168.31.150:22225 --rank 0 --world-size 2

machine 2 
python mmmg_mnist.py --init-method tcp://192.168.31.150:22225 --rank 1 --world-size 2
```
**参数说明：**     
```
--batch-size       type=int, default=1024, metavar='N', help='input batch size for training (default: 64)')
--test-batch-size  type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
--epochs           type=int, default=20, metavar='N',   help='number of epochs to train (default: 10)')
--lr               type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
--momentum         type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
--no-cuda          action='store_true', default=False,  help='disables CUDA training')
--seed             type=int, default=1, metavar='S', help='random seed (default: 1)')
--log-interval     type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
--init-method      type=str, default='tcp://127.0.0.1:23456'
--rank             type=int
--world-size       type=int
```
[machine 1 日志](pytorch-mmmg-log-5.md)    
[machine 2 日志](pytorch-mmmg-log-6.md)  
### 5.标准测试-3（mnist）
#### （1）.代码结构
```
-pytorch
--mmmg_data.py
```
#### 2.用法
```
machine 1 ip 192.168.31.150
python mmmg_data.py --init-method tcp://192.168.31.150:23456 --rank 0 --world-size 2   
machine 2 ip 192.168.31.170
python mmmg_data.py --init-method tcp://192.168.31.150:23456 --rank 1 --world-size 2  
```
**参数说明**：   
```
--backend         type=str,default='gloo',help='Name of the backend to use.'
-i --init-method  type=str,default='tcp://127.0.0.1:23456', help='URL specifying how to initialize the package.'
-r --rank         type=int, help='Rank of the current process.'
-s --world-size   type=int, help='Number of processes participating in the job.'
--epochs          type=int, default=20
--no-cuda         action='store_true'
--learning-rate -lr  type=float, default=1e-3
--root            type=str, default='../data'
--batch-size      type=int, default=128
```
[machine 1 log](pytorch-mmmg-log-7.md)   
[machine 2 log](pytorch-mmmg-log-8.md)    
------------------------------------------------------------------------------------------    
## 参考
[1] https://zhuanlan.zhihu.com/p/38949622    
[2] https://chenyue.top/2019/03/28/%E5%B7%A5%E7%A8%8B-%E5%9B%9B-Pytorch%E7%9A%84%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83/   
[3] https://pytorch.org/tutorials/intermediate/dist_tuto.html   
[4] https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group   
[5] https://github.com/narumiruna/pytorch-distributed-example   
[6] https://blog.csdn.net/m0_38008956/article/details/86559432   
[7] https://github.com/ShigekiKarita/pytorch-distributed-slurm-example   
[8] https://github.com/xhzhao/PyTorch-MPI-DDP-example   
[9] https://github.com/seba-1511/dist_tuto.pth  
[10] https://github.com/alexis-jacq/Pytorch-DPPO  
[11] https://blog.csdn.net/qq_20791919/article/details/79057648   
[12] https://blog.csdn.net/qq_20791919/article/details/79057871   
## 三、Docker环境测试
### 1.Docker环境准备
镜像：fusimeng/ai-pytorch:16.04-10.0-3.5-1.1.0   
### 2.数据准备
同上
### 3.代码准备
同上
### 4.测试
同上

