import numpy
import pandas as pd
import matplotlib.pyplot as plt

# 绘制标签等于num的肌电信号所有通道数据在一张图上
def plot_onelabel(data, label, num):
    index = []
    for i in range(data[:].shape[0]):
        if label[i] == num:
            index.append(i)
    # plt.axis([0, 200, -0.0005, 0.0005])
    plt.plot(data[index, :])
    plt.show()

def plot_channel(sig):
    for i in range(12):
        plt.subplot(12, 1, i + 1)
        plt.plot(sig[0:200, i])
        plt.yticks(fontsize=8)
        # plt.ylim(-0.0001, 0.0001)
    plt.show()



if __name__ == '__main__':
    for j in range(4, 5):
        root_data = 'F:/DB4'

        df = pd.read_hdf(root_data+'/data/restimulus/raw/DB4_s' + str(j) + 'raw.h5', 'df')
        df =numpy.array(df)
        plt.plot(df[40000:1050000, 13])
        plt.show()