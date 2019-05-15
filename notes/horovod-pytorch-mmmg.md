# Horovod + Pytorch 多机多卡标准测试
## 一、主机环境测试
### 1.两台主机
**===>>>硬件**   
|主机名|系统|IP|GPUs|
|:--:|:--:|:--:|:--:|  
|sdu3|Ubuntu16.04|192.168.199.53|1080ti * 4|
|sdu4|Ubuntu16.04|192.168.199.54|1080ti * 4|  
   
**===>>>系统**   
OS:[ubuntu16.04](https://github.com/fusimeng/ParallelComputing/blob/master/notes/serverinstall.md)    

**===>>>GPU Driver**   
[GPU Driver](https://github.com/fusimeng/ParallelComputing/blob/master/notes/driverinstall.md)   
  
**===>>>CUDA**   
[CUDA10.0 & cuDNN7.4](https://github.com/fusimeng/ParallelComputing/blob/master/notes/cudainstall.md)   
### 2.安装NCCL2
[参考链接](https://github.com/fusimeng/Horovod/blob/master/notes/install.md#1%E5%AE%89%E8%A3%85nccl-2)
### 3.安装GPUDirect
[参考链接](https://github.com/fusimeng/Horovod/blob/master/notes/install.md#2%E5%AE%89%E8%A3%85gpudirectoptional)
### 4.安装Openmpi
[参考链接](https://github.com/fusimeng/Horovod/blob/master/notes/install.md#3%E5%AE%89%E8%A3%85open-mpi)   
### 5.python环境(pytorch+horovod)
**===>>>安装Pytorch环境**   
[安装Anaconda](https://github.com/fusimeng/ai_tools),使用Anaconda，创建所需的环境   
* python3.6
* numpy
* pytorch 1.0.0
* torchvision 0.2.1
```shell
conda create --name pytorch python=3.6
source activate pytorch
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytorch torchvision
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

**===>>>[安装horovod环境](https://github.com/fusimeng/Horovod/blob/master/notes/install.md#4-horovodwith-pip)**    
### 6.代码结构 
```
-horovod
--pytorch_mnist.py
```
### 7.用法  目前还有问题？
主机-1（192.168.31.150）  
```
# horovodrun -np 2 -H 192.168.31.150:1,192.168.31.170:1 python pytorch_mnist.py
```
主机-2（192.168.31.170）  
```
# horovodrun -np 2 -H 192.168.31.150:1,192.168.31.170:1 python pytorch_mnist.py
```
[日志](../horovod/horovod_log2.md)

## 二、Docker环境
### 1.主机环境
在上述主机环境中安装docker。进行测试。   
`fusimeng/ai.horovod:v1`   
```
apt install net-tools
apt install iputils-ping
```
```
nvidia-docker run -itd -v /root/:/workspace --name=ff1 --hostname=ff1 fusimeng/ai.horovod:v1  
nvidia-docker run -itd -v /root/:/workspace --name=ff2 --hostname=ff2 fusimeng/ai.horovod:v1
```
### 2.配置免密登录   
(1) 在两台主机分别执行   
```
vim /etc/hosts  
写入以下内容：   
192.168.31.150  ff1
192.168.31.170  ff2
```
(2) 在两台主机分别执行   
```
ssh-keygen
ssh-copy-id [ip/hostname]
```
### 3.代码结构
同上
### 4.用法
