```
(pytorch) root@ff170:~/framework_benchmark/pytorch# python 3.py --init-method tcp://192.168.31.150:23456 --rank 1 --world-size 2
Namespace(backend='gloo', batch_size=128, epochs=20, init_method='tcp://192.168.31.150:23456', learning_rate=0.001, no_cuda=False, rank=1, root='../data', world_size=2)
Epoch: 1/20, train loss: 0.507103, train acc: 85.85%, test loss: 0.321021, test acc: 91.24%.
Epoch: 2/20, train loss: 0.313175, train acc: 91.20%, test loss: 0.290427, test acc: 91.77%.
Epoch: 3/20, train loss: 0.290227, train acc: 91.77%, test loss: 0.280378, test acc: 92.16%.
Epoch: 4/20, train loss: 0.279412, train acc: 92.09%, test loss: 0.275733, test acc: 92.33%.
Epoch: 5/20, train loss: 0.272866, train acc: 92.30%, test loss: 0.273204, test acc: 92.35%.
Epoch: 6/20, train loss: 0.268348, train acc: 92.45%, test loss: 0.271705, test acc: 92.42%.
Epoch: 7/20, train loss: 0.264959, train acc: 92.56%, test loss: 0.270775, test acc: 92.54%.
Epoch: 8/20, train loss: 0.262272, train acc: 92.66%, test loss: 0.270186, test acc: 92.61%.
Epoch: 9/20, train loss: 0.260054, train acc: 92.74%, test loss: 0.269817, test acc: 92.60%.
Epoch: 10/20, train loss: 0.258169, train acc: 92.79%, test loss: 0.269596, test acc: 92.64%.
Epoch: 11/20, train loss: 0.256531, train acc: 92.85%, test loss: 0.269481, test acc: 92.65%.
Epoch: 12/20, train loss: 0.255083, train acc: 92.90%, test loss: 0.269443, test acc: 92.61%.
Epoch: 13/20, train loss: 0.253787, train acc: 92.95%, test loss: 0.269464, test acc: 92.63%.
Epoch: 14/20, train loss: 0.252615, train acc: 92.97%, test loss: 0.269532, test acc: 92.64%.
Epoch: 15/20, train loss: 0.251546, train acc: 93.00%, test loss: 0.269636, test acc: 92.67%.
Epoch: 16/20, train loss: 0.250564, train acc: 93.03%, test loss: 0.269770, test acc: 92.68%.
Epoch: 17/20, train loss: 0.249656, train acc: 93.08%, test loss: 0.269928, test acc: 92.69%.
Epoch: 18/20, train loss: 0.248814, train acc: 93.12%, test loss: 0.270106, test acc: 92.69%.
Epoch: 19/20, train loss: 0.248029, train acc: 93.13%, test loss: 0.270301, test acc: 92.66%.
Epoch: 20/20, train loss: 0.247294, train acc: 93.16%, test loss: 0.270510, test acc: 92.66%.
```