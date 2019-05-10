# pytorch 多机多卡标准测试程序
## 一、环境介绍
### 1.两台主机
|主机名|系统|IP|GPUs|
|:--:|:--:|:--:|:--:|  
|ubuntu|Ubuntu16.04|192.168.31.150|Tesla P40 * 1|
|ff170|Ubuntu16.04|192.168.31.170|Tesla P40 * 1|
  
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
## 二、标准测试程序————1（MNIST）  
### 1.代码结构
```
-pytorch
--mnist.py
```
### 2.用法   
