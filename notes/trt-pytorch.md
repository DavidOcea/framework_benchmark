# 使用TensorRT5加速pytorch模型标准测试
通过pytorch搭建卷积神经网络完成手写识别任务，并将训练好的模型以多种方式部署到TensorRT中加速。
### 硬件环境   
```
nvidia-smi
Wed May  8 16:53:55 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.79       Driver Version: 410.79       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P40           Off  | 00000000:03:00.0 Off |                  Off |
| N/A   39C    P0    45W / 250W |      0MiB / 24451MiB |      3%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
## 一、主机环境
### 1.环境准备
(1) [安装Anaconda](https://github.com/fusimeng/ai_tools)    
(2) 使用Anaconda，创建所需的环境   
```shell
conda create --name pytorch python=3.6
source activate pytorch
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytorch torchvision tensorboardx pycuda
numpy

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
(3) 安装tensorrt   
[link](https://github.com/fusimeng/TensorRT/blob/master/notes/install.md)   
(4) 安装opencv   
[link](https://github.com/fusimeng/ParallelComputing/blob/master/notes/dockerai-2.md#2-%E4%B8%8B%E8%BD%BDopencv-410)   
### 2.数据准备
下载[mnist数据集](http://yann.lecun.com/exdb/mnist/)数据集，放在data目录下。   
### 3.代码准备       
``` 
-tensorrt 
--Config.py：卷积神经网络配置参数
--DataLoader.py：读取训练集与测试集
--Network.py：卷积神经网络结构
--OperateNetwork.py：对卷积神经网络的操作（训练，测试，保存读取权重，保存onnx）
--TensorRTNet.py：三种方式创建引擎
--main.py：主函数
```
```
注意：
(1)对于多输入模型保存onnx的方式：
dummy_input0 = torch.FloatTensor(Batch_size, seg_length).to(torch.device("cuda"))  
dummy_input1 = torch.FloatTensor(Batch_size, seg_length).to(torch.device("cuda"))  
dummy_input2 = torch.FloatTensor(Batch_size, seg_length).to(torch.device("cuda"))  
torch.onnx.export(model. (dummy_input0, dummy_input1, dummy_input2), filepath)  
(2)TensorRT不支持int64，float64,因此模型不应该包含这两种数据类型的运算。
```
### 4.测试及结果分析
**用法**：   
```shell
python smsg.py --lr 0.001 --epoch 1 --trainBatchSize 10000 --testBatchSize 10000 --num_workers 2 --log "../output/" 

optional arguments:   

--lr                default=1e-3    learning rate
--epoch             default=10     number of epochs tp train for
--trainBatchSize    default=100     training batch size
--testBatchSize     default=100     test batch size
--cuda              default=torch.cuda.is_available()  whether cuda is in use
--log               default="../output/"    storage logs/models
--num_workers       default=4, type=int,    number of workers to load data
--resume            default="../output/"    resume model 
备注： 模型加载，再次训练这个功能没有做
```
**pytorch指定显卡的几种方式**   
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
**代码用法**：   
同上
## Reference
[1] https://github.com/GuanLianzheng/pytorch_to_TensorRT5