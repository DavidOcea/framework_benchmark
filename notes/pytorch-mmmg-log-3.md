```
(pytorch) root@ubuntu:~/framework_benchmark/pytorch# python 1.py --rank 0 --world-size 2 --ip 192.168.31.150 --port 20001
Namespace(ip='192.168.31.150', port='20001', rank=0, world_size=2)
/root/anaconda3/envs/pytorch/lib/python3.6/site-packages/torch/distributed/distributed_c10d.py:86: UserWarning: torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead
  warnings.warn("torch.distributed.reduce_op is deprecated, please use "
rank: 0, step: 0, value: 5, reduced sum: 15.0.
rank: 0, step: 1, value: 5, reduced sum: 5.0.
rank: 0, step: 2, value: 0, reduced sum: 4.0.
rank: 0, step: 3, value: 0, reduced sum: 1.0.
rank: 0, step: 4, value: 8, reduced sum: 13.0.
rank: 0, step: 5, value: 5, reduced sum: 9.0.
rank: 0, step: 6, value: 6, reduced sum: 7.0.
rank: 0, step: 7, value: 6, reduced sum: 11.0.
rank: 0, step: 8, value: 7, reduced sum: 8.0.
rank: 0, step: 9, value: 1, reduced sum: 5.0.
rank: 0, step: 10, value: 6, reduced sum: 13.0.
rank: 0, step: 11, value: 1, reduced sum: 5.0.
rank: 0, step: 12, value: 0, reduced sum: 5.0.
rank: 0, step: 13, value: 10, reduced sum: 16.0.
rank: 0, step: 14, value: 8, reduced sum: 13.0.
rank: 0, step: 15, value: 3, reduced sum: 11.0.
rank: 0, step: 16, value: 8, reduced sum: 10.0.
rank: 0, step: 17, value: 9, reduced sum: 12.0.
rank: 0, step: 18, value: 1, reduced sum: 5.0.
rank: 0, step: 19, value: 9, reduced sum: 19.0.
rank: 0, step: 20, value: 9, reduced sum: 11.0.
rank: 0, step: 21, value: 3, reduced sum: 5.0.
rank: 0, step: 22, value: 9, reduced sum: 15.0.
rank: 0, step: 23, value: 4, reduced sum: 11.0.
rank: 0, step: 24, value: 2, reduced sum: 3.0.
rank: 0, step: 25, value: 7, reduced sum: 13.0.
rank: 0, step: 26, value: 7, reduced sum: 10.0.
rank: 0, step: 27, value: 7, reduced sum: 10.0.
rank: 0, step: 28, value: 9, reduced sum: 16.0.
rank: 0, step: 29, value: 2, reduced sum: 9.0.
rank: 0, step: 30, value: 4, reduced sum: 13.0.
rank: 0, step: 31, value: 1, reduced sum: 4.0.
rank: 0, step: 32, value: 9, reduced sum: 12.0.
rank: 0, step: 33, value: 4, reduced sum: 13.0.
rank: 0, step: 34, value: 3, reduced sum: 6.0.
rank: 0, step: 35, value: 8, reduced sum: 12.0.
rank: 0, step: 36, value: 5, reduced sum: 12.0.
rank: 0, step: 37, value: 4, reduced sum: 13.0.
rank: 0, step: 38, value: 1, reduced sum: 8.0.
rank: 0, step: 39, value: 10, reduced sum: 19.0.
rank: 0, step: 40, value: 2, reduced sum: 10.0.
rank: 0, step: 41, value: 8, reduced sum: 11.0.
rank: 0, step: 42, value: 0, reduced sum: 4.0.
rank: 0, step: 43, value: 0, reduced sum: 0.0.
rank: 0, step: 44, value: 2, reduced sum: 10.0.
rank: 0, step: 45, value: 5, reduced sum: 12.0.
rank: 0, step: 46, value: 8, reduced sum: 14.0.
rank: 0, step: 47, value: 0, reduced sum: 2.0.
rank: 0, step: 48, value: 6, reduced sum: 14.0.
rank: 0, step: 49, value: 4, reduced sum: 6.0.
rank: 0, step: 50, value: 1, reduced sum: 5.0.
rank: 0, step: 51, value: 0, reduced sum: 9.0.
rank: 0, step: 52, value: 5, reduced sum: 10.0.
rank: 0, step: 53, value: 10, reduced sum: 10.0.
rank: 0, step: 54, value: 3, reduced sum: 6.0.
rank: 0, step: 55, value: 0, reduced sum: 9.0.
rank: 0, step: 56, value: 3, reduced sum: 13.0.
rank: 0, step: 57, value: 9, reduced sum: 10.0.
rank: 0, step: 58, value: 1, reduced sum: 6.0.
rank: 0, step: 59, value: 10, reduced sum: 13.0.
rank: 0, step: 60, value: 3, reduced sum: 10.0.
rank: 0, step: 61, value: 3, reduced sum: 4.0.
rank: 0, step: 62, value: 2, reduced sum: 3.0.
rank: 0, step: 63, value: 1, reduced sum: 8.0.
rank: 0, step: 64, value: 9, reduced sum: 11.0.
rank: 0, step: 65, value: 2, reduced sum: 7.0.
rank: 0, step: 66, value: 4, reduced sum: 11.0.
rank: 0, step: 67, value: 7, reduced sum: 9.0.
rank: 0, step: 68, value: 8, reduced sum: 12.0.
rank: 0, step: 69, value: 7, reduced sum: 15.0.
rank: 0, step: 70, value: 8, reduced sum: 12.0.
rank: 0, step: 71, value: 6, reduced sum: 16.0.
rank: 0, step: 72, value: 0, reduced sum: 6.0.
rank: 0, step: 73, value: 2, reduced sum: 7.0.
rank: 0, step: 74, value: 1, reduced sum: 6.0.
rank: 0, step: 75, value: 3, reduced sum: 11.0.
rank: 0, step: 76, value: 6, reduced sum: 11.0.
rank: 0, step: 77, value: 3, reduced sum: 11.0.
rank: 0, step: 78, value: 2, reduced sum: 7.0.
rank: 0, step: 79, value: 2, reduced sum: 7.0.
rank: 0, step: 80, value: 4, reduced sum: 10.0.
rank: 0, step: 81, value: 2, reduced sum: 6.0.
rank: 0, step: 82, value: 10, reduced sum: 10.0.
rank: 0, step: 83, value: 5, reduced sum: 14.0.
rank: 0, step: 84, value: 2, reduced sum: 2.0.
rank: 0, step: 85, value: 8, reduced sum: 16.0.
rank: 0, step: 86, value: 4, reduced sum: 7.0.
rank: 0, step: 87, value: 1, reduced sum: 8.0.
rank: 0, step: 88, value: 7, reduced sum: 10.0.
rank: 0, step: 89, value: 8, reduced sum: 18.0.
rank: 0, step: 90, value: 5, reduced sum: 10.0.
rank: 0, step: 91, value: 2, reduced sum: 9.0.
rank: 0, step: 92, value: 8, reduced sum: 10.0.
rank: 0, step: 93, value: 1, reduced sum: 7.0.
rank: 0, step: 94, value: 1, reduced sum: 9.0.
rank: 0, step: 95, value: 1, reduced sum: 5.0.
rank: 0, step: 96, value: 4, reduced sum: 5.0.
rank: 0, step: 97, value: 3, reduced sum: 8.0.
rank: 0, step: 98, value: 3, reduced sum: 7.0.
rank: 0, step: 99, value: 7, reduced sum: 16.0.
```