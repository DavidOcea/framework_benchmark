```
(pytorch) root@ubuntu:~/framework_benchmark/pytorch# python 2.py --init-method tcp://192.168.31.150:22225 --rank 0 --world-size 2
2.py:85: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.355105
Train Epoch: 1 [10240/60000 (17%)]	Loss: 2.305522
Train Epoch: 1 [20480/60000 (34%)]	Loss: 2.303816
Train Epoch: 1 [30720/60000 (51%)]	Loss: 2.278314
Train Epoch: 1 [40960/60000 (68%)]	Loss: 2.254969
Train Epoch: 1 [51200/60000 (85%)]	Loss: 2.236166
Epoch 1 of 20 took 10.533s
2.py:120: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  data, target = Variable(data, volatile=True), Variable(target)
/root/anaconda3/envs/pytorch/lib/python3.6/site-packages/torch/nn/_reduction.py:49: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))

Test set: Average loss: 2.1882, Accuracy: 3489/10000 (34%)

Train Epoch: 2 [0/60000 (0%)]	Loss: 2.211128
Train Epoch: 2 [10240/60000 (17%)]	Loss: 2.172995
Train Epoch: 2 [20480/60000 (34%)]	Loss: 2.124129
Train Epoch: 2 [30720/60000 (51%)]	Loss: 2.070790
Train Epoch: 2 [40960/60000 (68%)]	Loss: 1.967799
Train Epoch: 2 [51200/60000 (85%)]	Loss: 1.865286
Epoch 2 of 20 took 9.602s

Test set: Average loss: 1.6010, Accuracy: 6058/10000 (60%)

Train Epoch: 3 [0/60000 (0%)]	Loss: 1.849395
Train Epoch: 3 [10240/60000 (17%)]	Loss: 1.676990
Train Epoch: 3 [20480/60000 (34%)]	Loss: 1.574988
Train Epoch: 3 [30720/60000 (51%)]	Loss: 1.547190
Train Epoch: 3 [40960/60000 (68%)]	Loss: 1.430171
Train Epoch: 3 [51200/60000 (85%)]	Loss: 1.312671
Epoch 3 of 20 took 9.582s

Test set: Average loss: 0.9074, Accuracy: 8077/10000 (80%)

Train Epoch: 4 [0/60000 (0%)]	Loss: 1.289176
Train Epoch: 4 [10240/60000 (17%)]	Loss: 1.218667
Train Epoch: 4 [20480/60000 (34%)]	Loss: 1.138543
Train Epoch: 4 [30720/60000 (51%)]	Loss: 1.121094
Train Epoch: 4 [40960/60000 (68%)]	Loss: 1.061772
Train Epoch: 4 [51200/60000 (85%)]	Loss: 1.005182
Epoch 4 of 20 took 9.591s

Test set: Average loss: 0.5713, Accuracy: 8613/10000 (86%)

Train Epoch: 5 [0/60000 (0%)]	Loss: 0.959331
Train Epoch: 5 [10240/60000 (17%)]	Loss: 0.920527
Train Epoch: 5 [20480/60000 (34%)]	Loss: 0.936621
Train Epoch: 5 [30720/60000 (51%)]	Loss: 0.846621
Train Epoch: 5 [40960/60000 (68%)]	Loss: 0.856829
Train Epoch: 5 [51200/60000 (85%)]	Loss: 0.831947
Epoch 5 of 20 took 9.627s

Test set: Average loss: 0.4550, Accuracy: 8817/10000 (88%)

Train Epoch: 6 [0/60000 (0%)]	Loss: 0.821289
Train Epoch: 6 [10240/60000 (17%)]	Loss: 0.766848
Train Epoch: 6 [20480/60000 (34%)]	Loss: 0.784586
Train Epoch: 6 [30720/60000 (51%)]	Loss: 0.800820
Train Epoch: 6 [40960/60000 (68%)]	Loss: 0.705075
Train Epoch: 6 [51200/60000 (85%)]	Loss: 0.739847
Epoch 6 of 20 took 9.795s

Test set: Average loss: 0.3865, Accuracy: 8925/10000 (89%)

Train Epoch: 7 [0/60000 (0%)]	Loss: 0.786049
Train Epoch: 7 [10240/60000 (17%)]	Loss: 0.691403
Train Epoch: 7 [20480/60000 (34%)]	Loss: 0.692824
Train Epoch: 7 [30720/60000 (51%)]	Loss: 0.725488
Train Epoch: 7 [40960/60000 (68%)]	Loss: 0.653926
Train Epoch: 7 [51200/60000 (85%)]	Loss: 0.649727
Epoch 7 of 20 took 9.595s

Test set: Average loss: 0.3474, Accuracy: 9046/10000 (90%)

Train Epoch: 8 [0/60000 (0%)]	Loss: 0.654028
Train Epoch: 8 [10240/60000 (17%)]	Loss: 0.667873
Train Epoch: 8 [20480/60000 (34%)]	Loss: 0.650010
Train Epoch: 8 [30720/60000 (51%)]	Loss: 0.596245
Train Epoch: 8 [40960/60000 (68%)]	Loss: 0.651126
Train Epoch: 8 [51200/60000 (85%)]	Loss: 0.677375
Epoch 8 of 20 took 9.657s

Test set: Average loss: 0.3129, Accuracy: 9127/10000 (91%)

Train Epoch: 9 [0/60000 (0%)]	Loss: 0.600588
Train Epoch: 9 [10240/60000 (17%)]	Loss: 0.581434
Train Epoch: 9 [20480/60000 (34%)]	Loss: 0.598328
Train Epoch: 9 [30720/60000 (51%)]	Loss: 0.610360
Train Epoch: 9 [40960/60000 (68%)]	Loss: 0.595473
Train Epoch: 9 [51200/60000 (85%)]	Loss: 0.555459
Epoch 9 of 20 took 9.609s

Test set: Average loss: 0.2868, Accuracy: 9191/10000 (91%)

Train Epoch: 10 [0/60000 (0%)]	Loss: 0.535823
Train Epoch: 10 [10240/60000 (17%)]	Loss: 0.564952
Train Epoch: 10 [20480/60000 (34%)]	Loss: 0.581805
Train Epoch: 10 [30720/60000 (51%)]	Loss: 0.564084
Train Epoch: 10 [40960/60000 (68%)]	Loss: 0.524607
Train Epoch: 10 [51200/60000 (85%)]	Loss: 0.561403
Epoch 10 of 20 took 9.656s

Test set: Average loss: 0.2648, Accuracy: 9238/10000 (92%)

Train Epoch: 11 [0/60000 (0%)]	Loss: 0.558024
Train Epoch: 11 [10240/60000 (17%)]	Loss: 0.500761
Train Epoch: 11 [20480/60000 (34%)]	Loss: 0.541054
Train Epoch: 11 [30720/60000 (51%)]	Loss: 0.565119
Train Epoch: 11 [40960/60000 (68%)]	Loss: 0.552010
Train Epoch: 11 [51200/60000 (85%)]	Loss: 0.566469
Epoch 11 of 20 took 9.311s

Test set: Average loss: 0.2501, Accuracy: 9281/10000 (92%)

Train Epoch: 12 [0/60000 (0%)]	Loss: 0.495560
Train Epoch: 12 [10240/60000 (17%)]	Loss: 0.480602
Train Epoch: 12 [20480/60000 (34%)]	Loss: 0.541533
Train Epoch: 12 [30720/60000 (51%)]	Loss: 0.484270
Train Epoch: 12 [40960/60000 (68%)]	Loss: 0.545176
Train Epoch: 12 [51200/60000 (85%)]	Loss: 0.523254
Epoch 12 of 20 took 9.223s

Test set: Average loss: 0.2365, Accuracy: 9317/10000 (93%)

Train Epoch: 13 [0/60000 (0%)]	Loss: 0.442988
Train Epoch: 13 [10240/60000 (17%)]	Loss: 0.472208
Train Epoch: 13 [20480/60000 (34%)]	Loss: 0.506856
Train Epoch: 13 [30720/60000 (51%)]	Loss: 0.538371
Train Epoch: 13 [40960/60000 (68%)]	Loss: 0.531676
Train Epoch: 13 [51200/60000 (85%)]	Loss: 0.463722
Epoch 13 of 20 took 9.256s

Test set: Average loss: 0.2224, Accuracy: 9346/10000 (93%)

Train Epoch: 14 [0/60000 (0%)]	Loss: 0.500128
Train Epoch: 14 [10240/60000 (17%)]	Loss: 0.409345
Train Epoch: 14 [20480/60000 (34%)]	Loss: 0.483233
Train Epoch: 14 [30720/60000 (51%)]	Loss: 0.478735
Train Epoch: 14 [40960/60000 (68%)]	Loss: 0.452786
Train Epoch: 14 [51200/60000 (85%)]	Loss: 0.466303
Epoch 14 of 20 took 9.408s

Test set: Average loss: 0.2091, Accuracy: 9388/10000 (93%)

Train Epoch: 15 [0/60000 (0%)]	Loss: 0.433427
Train Epoch: 15 [10240/60000 (17%)]	Loss: 0.483422
Train Epoch: 15 [20480/60000 (34%)]	Loss: 0.439088
Train Epoch: 15 [30720/60000 (51%)]	Loss: 0.421583
Train Epoch: 15 [40960/60000 (68%)]	Loss: 0.432021
Train Epoch: 15 [51200/60000 (85%)]	Loss: 0.433395
Epoch 15 of 20 took 9.238s

Test set: Average loss: 0.2014, Accuracy: 9416/10000 (94%)

Train Epoch: 16 [0/60000 (0%)]	Loss: 0.392646
Train Epoch: 16 [10240/60000 (17%)]	Loss: 0.416882
Train Epoch: 16 [20480/60000 (34%)]	Loss: 0.461630
Train Epoch: 16 [30720/60000 (51%)]	Loss: 0.440798
Train Epoch: 16 [40960/60000 (68%)]	Loss: 0.452248
Train Epoch: 16 [51200/60000 (85%)]	Loss: 0.413541
Epoch 16 of 20 took 9.657s

Test set: Average loss: 0.1909, Accuracy: 9419/10000 (94%)

Train Epoch: 17 [0/60000 (0%)]	Loss: 0.437764
Train Epoch: 17 [10240/60000 (17%)]	Loss: 0.482642
Train Epoch: 17 [20480/60000 (34%)]	Loss: 0.432014
Train Epoch: 17 [30720/60000 (51%)]	Loss: 0.405187
Train Epoch: 17 [40960/60000 (68%)]	Loss: 0.368783
Train Epoch: 17 [51200/60000 (85%)]	Loss: 0.402148
Epoch 17 of 20 took 9.640s

Test set: Average loss: 0.1817, Accuracy: 9460/10000 (94%)

Train Epoch: 18 [0/60000 (0%)]	Loss: 0.400885
Train Epoch: 18 [10240/60000 (17%)]	Loss: 0.430700
Train Epoch: 18 [20480/60000 (34%)]	Loss: 0.436492
Train Epoch: 18 [30720/60000 (51%)]	Loss: 0.377381
Train Epoch: 18 [40960/60000 (68%)]	Loss: 0.378874
Train Epoch: 18 [51200/60000 (85%)]	Loss: 0.390412
Epoch 18 of 20 took 9.573s

Test set: Average loss: 0.1782, Accuracy: 9467/10000 (94%)

Train Epoch: 19 [0/60000 (0%)]	Loss: 0.396091
Train Epoch: 19 [10240/60000 (17%)]	Loss: 0.377676
Train Epoch: 19 [20480/60000 (34%)]	Loss: 0.408569
Train Epoch: 19 [30720/60000 (51%)]	Loss: 0.363412
Train Epoch: 19 [40960/60000 (68%)]	Loss: 0.334436
Train Epoch: 19 [51200/60000 (85%)]	Loss: 0.389399
Epoch 19 of 20 took 9.302s

Test set: Average loss: 0.1694, Accuracy: 9486/10000 (94%)

Train Epoch: 20 [0/60000 (0%)]	Loss: 0.409568
Train Epoch: 20 [10240/60000 (17%)]	Loss: 0.382378
Train Epoch: 20 [20480/60000 (34%)]	Loss: 0.402718
Train Epoch: 20 [30720/60000 (51%)]	Loss: 0.364220
Train Epoch: 20 [40960/60000 (68%)]	Loss: 0.375730
Train Epoch: 20 [51200/60000 (85%)]	Loss: 0.370848
Epoch 20 of 20 took 9.235s

Test set: Average loss: 0.1633, Accuracy: 9498/10000 (94%)

Total time= 191.090s
```