# Pytorch TensorBoard标准测试程序

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
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboardx tensorflow(cpu版即可)
```
### 2.数据准备
目前的代码示例不需要，如需复杂的示例，请到官方GitHub查看。   
### 3.代码准备       
``` 
-tensorboard  
--01_scalar_base.py
--02_graph_model.py
--03_linear_regression.py
--demo.py

```
### 4.测试
**用法**：   
```shell
python 01_scalar_base.py 
or
python 02_graph_model.py
or
python 03_linear_regression.py
or demo.py

然后
tensorboard --logdir xxx
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
同上

-----
# Reference
[1] https://www.jianshu.com/p/46eb3004beca