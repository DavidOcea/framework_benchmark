```
(pytorch) root@ubuntu:~/framework_benchmark/pytorch# python 3.py --init-method tcp://192.168.31.150:23456 --rank 0 --world-size 2
Namespace(backend='gloo', batch_size=128, epochs=20, init_method='tcp://192.168.31.150:23456', learning_rate=0.001, no_cuda=False, rank=0, root='../data', world_size=2)
Epoch: 1/20, train loss: 0.512819, train acc: 85.58%, test loss: 0.321021, test acc: 91.24%.
Epoch: 2/20, train loss: 0.317882, train acc: 91.01%, test loss: 0.290427, test acc: 91.77%.
Epoch: 3/20, train loss: 0.294605, train acc: 91.71%, test loss: 0.280378, test acc: 92.16%.
Epoch: 4/20, train loss: 0.283352, train acc: 92.06%, test loss: 0.275733, test acc: 92.33%.
Epoch: 5/20, train loss: 0.276316, train acc: 92.26%, test loss: 0.273204, test acc: 92.35%.
Epoch: 6/20, train loss: 0.271322, train acc: 92.47%, test loss: 0.271705, test acc: 92.42%.
Epoch: 7/20, train loss: 0.267508, train acc: 92.57%, test loss: 0.270775, test acc: 92.54%.
Epoch: 8/20, train loss: 0.264453, train acc: 92.67%, test loss: 0.270186, test acc: 92.61%.
Epoch: 9/20, train loss: 0.261924, train acc: 92.74%, test loss: 0.269817, test acc: 92.60%.
Epoch: 10/20, train loss: 0.259777, train acc: 92.80%, test loss: 0.269596, test acc: 92.64%.
Epoch: 11/20, train loss: 0.257918, train acc: 92.89%, test loss: 0.269481, test acc: 92.65%.
Epoch: 12/20, train loss: 0.256285, train acc: 92.95%, test loss: 0.269443, test acc: 92.61%.
Epoch: 13/20, train loss: 0.254833, train acc: 92.99%, test loss: 0.269464, test acc: 92.63%.
Epoch: 14/20, train loss: 0.253528, train acc: 93.02%, test loss: 0.269532, test acc: 92.64%.
Epoch: 15/20, train loss: 0.252345, train acc: 93.03%, test loss: 0.269636, test acc: 92.67%.
Epoch: 16/20, train loss: 0.251266, train acc: 93.05%, test loss: 0.269770, test acc: 92.68%.
Epoch: 17/20, train loss: 0.250275, train acc: 93.10%, test loss: 0.269928, test acc: 92.69%.
Epoch: 18/20, train loss: 0.249360, train acc: 93.11%, test loss: 0.270106, test acc: 92.69%.
Epoch: 19/20, train loss: 0.248512, train acc: 93.15%, test loss: 0.270301, test acc: 92.66%.
Epoch: 20/20, train loss: 0.247722, train acc: 93.15%, test loss: 0.270510, test acc: 92.66%.

```