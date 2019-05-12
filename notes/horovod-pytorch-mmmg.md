# Horovod + Pytorch 多机多卡标准测试
## 一、主机环境测试
### 1.两台主机
|主机名|系统|IP|GPUs|
|:--:|:--:|:--:|:--:|  
|ubuntu|Ubuntu16.04|192.168.31.150|Tesla P40 * 1|
|ff170|Ubuntu16.04|192.168.31.170|Tesla P40 * 1|

OS:ubuntu16.04   
CUDA:10.0   
cuDNN：7.4   

machine-1:  
```
(pytorch) root@ubuntu:~/package/horovod# nvidia-smi
Sun May 12 19:05:51 2019       
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
machine-2:   
```
(pytorch) root@ff170:~/package/nccl-tests# nvidia-smi
Sun May 12 19:06:28 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P40           Off  | 00000000:03:00.0 Off |                    0 |
| N/A   35C    P0    44W / 250W |      0MiB / 22919MiB |      4%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
### 2.python环境
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

### 3.安装horovod环境
参考：https://github.com/fusimeng/Horovod/blob/master/notes/install.md   
### 4.配置免密登录

### 5.代码结构 
```
-horovod
--pytorch_mnist.py
```

### 6.用法
```
$ horovodrun -np 4 -H localhost:4 python pytorch_mnist.py
```
[日志](../horovod/horovod_log2.md)

## 二、Docker环境
