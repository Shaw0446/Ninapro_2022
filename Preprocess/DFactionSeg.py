import math
import h5py
import pandas as pd
import numpy as np
import tsaug
from sklearn import preprocessing
import scipy.signal as signal
import nina_funcs as nf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
# gestures = list(range(1, 50))
dir='F:/DB2'
'''
    本类将上一个实验的预处理部分用dataframe格式重写,便于后续实验在
    数据处理上共用,也是更符合github工具包的存储格式
'''

'''  df形式的动作分割（要求能不排除休息状态）'''
def action_seg(data, channel, endlabel):
    actionList = []  # 存储动作
    begin = 0
    for aim in tqdm(range(len(data) - 1)):
        # 控制手势停止时间
        if (data.iloc[aim, channel] == endlabel + 1):
            break;
        if (aim == len(data) - 1 or data.iloc[aim, channel] != data.iloc[aim + 1, 12]):
            end = aim
            actionList.append(data[begin:end])
            begin = end + 1
    return actionList

'''数据标准化（list的元素为df的方式）'''
def uniform(actionList, channel):
    for action in actionList:
        iemg = action.iloc[:, :channel].copy()
        scaler = preprocessing.StandardScaler()
        BNemg = scaler.fit_transform(iemg)
        action = action.copy(False)
        action.iloc[:, :channel] = BNemg

    return actionList


'''
    该函数在输入二维数组时会将各个通道随机打乱，破坏了不同电极采样的同步性，故设定随机种子，单个通道依次打乱
        输入的数据是一维列向量，新生成的数据是一维行向量
'''
def OneEnhanceEMG(data):
    newdata =tsaug.TimeWarp(n_speed_change=1, max_speed_ratio=1.5, seed=123).augment(data)
#     enhancedata =np.hstack((data, newdata))
    return data, newdata



'''
    将数据增强写在标准化里面是为了将增强的数据和原始数据分开标准化，拼接的标准化方法会影响原始数据。
    不分割为肌电子图返回数据标准化后结果,BNdatalist元素为单独一个动作手势标准化后的结果，
    保存有对应的标签 ,返回类型(time,channel),考虑到实验流程，把数据增强放在标准化前
'''
def bnEnhancesegment(seglist, channel=12):
    channel = 12
    BNdatalist = []
    for i in range(len(seglist)):
        temp_label = seglist[i].iloc[0, channel]
        temp_rep = seglist[i].iloc[0, channel+1]
        #跳过标签为0
        if(temp_label==0):
           continue
        if temp_rep in([1, 3, 4, 6]):
            # 增强前后的手势分开标准化，对训练集第1，3，4，6次进行增强
            timedata = np.array(seglist[i].iloc[:, :channel].copy())
            data = []
            newdata = []
            for Numchannel in range(12):
                datasample, newdatasample = OneEnhanceEMG(timedata[:, Numchannel])
                data.append(datasample)
                newdata.append(newdatasample)
            data = (np.array(data)).T       #将数据转换为（time,channel）
            newdata = (np.array(newdata)).T
            # flag来控制对未增强数据的保存
            flag = True
            for iemg in ([data, newdata]):
                scaler = StandardScaler()
                Zscdata = scaler.fit_transform(iemg[:])
                BNdata = Zscdata
                if (flag):
                    bndata1 =BNdata
                    flag=False
                else:
                    bndata2 =BNdata
            #数据倍增后，标签倍增
            BNdata = np.vstack((bndata1, bndata2))
            BNlabel = np.hstack([seglist[i].iloc[:, channel], seglist[i].iloc[:, channel]])
            BNrep = np.hstack([seglist[i].iloc[:, channel+1], seglist[i].iloc[:, channel+1]])
        #测试集不做数据增强
        else:
            iemg = np.array(seglist[i].iloc[:, :channel].copy())
            scaler = StandardScaler()
            Zscdata = scaler.fit_transform(iemg[:])
            BNdata = Zscdata
            BNlabel= np.array(seglist[i].iloc[:, channel])
            BNrep = np.array(seglist[i].iloc[:, channel+1])
        myemg=pd.DataFrame(BNdata)
        myemg['stimulus'] = BNlabel
        myemg['repetition'] = BNrep
        BNdatalist.append(myemg)
    return BNdatalist





'''时间增量窗口分割和划分数据集，输入参数为存储动作的list'''
def action_comb(actionList, timeWindow, strideWindow, channel=12):
    emgList = [[], [], [], [], [], [], []]
    labelList = [[], [], [], [], [], [], []]
    for action in actionList:
        rep = int(action.values[0, channel + 1])
        if rep == 0:        #暂时不考虑休息状态
            continue
        length = math.floor((len(action) - timeWindow) / strideWindow) + 1
        for j in range(length):
            subImage = action.iloc[strideWindow * j:strideWindow * j + timeWindow, 0:channel]  # 连续分割方式
            emgList[rep].append(subImage)
            labelList[rep].append(action.iloc[0, channel])

    for i in range(len(emgList)):
        emgList[i] = np.array(emgList[i])
    for i in range(len(labelList)):
        labelList[i] = np.array(labelList[i])
    return emgList, labelList


for j in range(1, 2):
    df = pd.read_hdf(dir+'/data/filter/DB2_s' + str(j) + 'filter.h5', 'df')

    '''滑动窗口分割'''
    actionList = action_seg(df, 12, 49)
    # unList = uniform(actionList, 12)
    bnlist=bnEnhancesegment(actionList)
    emgList, labelList = action_comb(bnlist, 400, 100)
    # # 存储为h5文件
    file = h5py.File(dir+'/lastdata/Seg/DB2_s' + str(j) + 'Seg.h5', 'w')
    file.create_dataset('Data1', data=(emgList[1]).astype('float32'))
    file.create_dataset('Data2', data=(emgList[2]).astype('float32'))
    file.create_dataset('Data3', data=(emgList[3]).astype('float32'))
    file.create_dataset('Data4', data=(emgList[4]).astype('float32'))
    file.create_dataset('Data5', data=(emgList[5]).astype('float32'))
    file.create_dataset('Data6', data=(emgList[6]).astype('float32'))
    file.create_dataset('label1', data=(labelList[1]).astype('int'))
    file.create_dataset('label2', data=(labelList[2]).astype('int'))
    file.create_dataset('label3', data=(labelList[3]).astype('int'))
    file.create_dataset('label4', data=(labelList[4]).astype('int'))
    file.create_dataset('label5', data=(labelList[5]).astype('int'))
    file.create_dataset('label6', data=(labelList[6]).astype('int'))
    file.close()
    print('******************DB2_s' + str(j) + '分割完成***********************')

