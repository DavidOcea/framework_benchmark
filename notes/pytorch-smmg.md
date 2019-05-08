# pytorch单机多卡标准测试 
### 硬件环境  
```
nvidia-smi
Wed May  8 16:54:50 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.78       Driver Version: 410.78       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 208...  Off  | 00000000:04:00.0 Off |                  N/A |
| 22%   36C    P0    64W / 250W |      0MiB / 10989MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce RTX 208...  Off  | 00000000:05:00.0 Off |                  N/A |
| 22%   38C    P0    68W / 250W |      0MiB / 10989MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce RTX 208...  Off  | 00000000:06:00.0 Off |                  N/A |
| 22%   37C    P0    62W / 250W |      0MiB / 10989MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  GeForce RTX 208...  Off  | 00000000:07:00.0 Off |                  N/A |
| 22%   37C    P0    59W / 250W |      0MiB / 10989MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   4  GeForce RTX 208...  Off  | 00000000:08:00.0 Off |                  N/A |
| 23%   37C    P0    61W / 250W |      0MiB / 10989MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   5  GeForce RTX 208...  Off  | 00000000:0B:00.0 Off |                  N/A |
| 23%   37C    P0    62W / 250W |      0MiB / 10989MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   6  GeForce RTX 208...  Off  | 00000000:0C:00.0 Off |                  N/A |
| 20%   38C    P0    55W / 250W |      0MiB / 10989MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+
|   7  GeForce RTX 208...  Off  | 00000000:0D:00.0 Off |                  N/A |
|  7%   37C    P0    54W / 250W |      0MiB / 10989MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+
|   8  GeForce RTX 208...  Off  | 00000000:0E:00.0 Off |                  N/A |
| 34%   43C    P0    71W / 250W |      0MiB / 10989MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   9  GeForce RTX 208...  Off  | 00000000:0F:00.0 Off |                  N/A |
| 50%   79C    P2   243W / 250W |  10629MiB / 10989MiB |     90%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    9     22113      C   python                                     10619MiB |
+-----------------------------------------------------------------------------+
```
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
（我使用的）   
conda install pytorch torchvision 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  tensorboardx
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
**代码2目录介绍**：   
使用torch.nn.DataParallel方法。    
参考：https://github.com/dnddnjs/pytorch-multigpu       
``` 
-pytorch  # pytorch标准测试代码目录 
--model.py  # 模型
--smsg2.py # 测试主程序
```
 
### 4.测试及结果分析
**代码2用法**：         
```shell
python smmg2.py --gpu_devices 0 1 2 3 --batch_size 768
```
optional arguments:   
```
--resume            default=None    
--batch_size        default=768
--num_worker        default=4
--gpu_devices       default=None
--lr                default=1e-3    learning rate
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
