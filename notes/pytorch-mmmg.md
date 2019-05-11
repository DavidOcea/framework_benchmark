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
## 二、标准测试程序————1（toy）  
### 1.代码结构
```
-pytorch
--mmmg_toy.py
```
### 2.用法   
**在单机中使用**   
```
Terminal 1
$ python mmmg_toy.py --rank 0 --world-size 2
Terminal 2
$ python mmmg_toy.py --rank 1 --world-size 2
```
[Terminal 1 的日志](terminal1.md)   
[Terminal 2 的日志](terminal2.md)
   
**在多机中使用**   
```
Machine 1 with ip 192.168.31.150

$ python mmmg_toy.py --rank 0 --world-size 2 --ip 192.168.31.150 --port 22000
Machine 2

$ python mmmg_toy.py --rank 1 --world-size 2 --ip 192.168.31.150 --port 22000
```
[Machine 1 的日志](log1.md)   
[Machine 2 的日志](log2.md)   
## 三、标准测试程序————2（mnist）
### 1.代码结构
```
-pytorch
--mmmg_mnist.py
```
### 2.用法
```
machine 1 ip 192.168.31.150 
python mmmg_mnist.py --init-method tcp://192.168.31.150:22225 --rank 0 --world-size 2

machine 2 
python mmmg_mnist.py --init-method tcp://192.168.31.150:22225 --rank 1 --world-size 2
```
参数说明：   
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
[machine 1 日志](log3.md)    
[machine 2 日志](log4.md)  
## 四、标准程序测试————3（mnist）
### 1.代码结构
```
-pytorch
--mmmg_data.py
```
### 2.用法
```
machine 1 ip 192.168.31.150
python 3.py --init-method tcp://192.168.31.150:23456 --rank 0 --world-size 2   
machine 2 ip 192.168.31.170
python 3.py --init-method tcp://192.168.31.150:23456 --rank 1 --world-size 2  
```
用法：   
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
[machine 1 log](log5.md)   
[machine 2 log](log6.md)    
**------------------------------------------------------------------------------------------**    
# 参考
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


