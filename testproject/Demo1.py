import h5py
import numpy as np

# v = 0.0000000000000009876543212133
# a=float('%.6g' %v)
# print("v=",v)
# print("a=",a)
#
# b=a
# print(float(b))


from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
# 产生一个测试信号，振幅为2的正弦波，其频率在3kHZ缓慢调制，振幅以指数形式下降的白噪声
for j in range(1,7):
    print(j)