import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nina_funcs as nf
from sklearn import preprocessing

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
gestures = list(range(1,50))
dir='F:/DB2'

def fea_transpose(data):
    # 为了便于归一化，对矩阵进行转置
    arr_T = np.transpose(np.array(data))
    sc_fea = preprocessing.StandardScaler().fit_transform(arr_T)
    arr_fea = np.transpose(sc_fea)
    return arr_fea




for j in range(1, 2):
    file = h5py.File(dir+'/data/Comb_downSeg/DB2_s' + str(j) + 'Seg17.h5', 'r')
    '''step1: 数据集划分'''
    x_train1, x_train3, x_train4, x_train6 = file['x_train1'][:], file['x_train3'][:], file['x_train4'][:], file['x_train6'][:]
    y_train1, y_train3, y_train4, y_train6 = file['y_train1'][:], file['y_train3'][:], file['y_train4'][:], file['y_train6'][:]
    x_test = file['x_test'][:]
    y_test = file['y_test'][:]
    file.close()

    x_train = np.concatenate([x_train1, x_train3, x_train4, x_train6], axis=0)
    y_train = np.concatenate([y_train1, y_train3, y_train4, y_train6], axis=0)

    '''step2: 选择特征组合和归一化'''
    # 选择特征组合
    features = [nf.rms,nf.min,nf.max]
    # features = [nf.emg_dwpt, nf.iemg,nf.rms,nf.hist,nf.entropy,nf.kurtosis,nf.zero_cross,nf.min,nf.max,nf.mean,nf.median,nf.psd]
    temp = x_train[:, :, :]
    train_feature = nf.feature_extractor(features=features, shape=(temp.shape[0], -1), data=temp)
    test_feature = nf.feature_extractor(features, (x_test.shape[0], -1), x_test)

    # ss = preprocessing.Normalizer(norm="l2")
    # ss.fit(train_feature)
    # sc_train = ss.transform(train_feature)
    # sc_test = ss.transform(test_feature)

    # x_test_fea = sc_test .reshape(sc_test.shape[0], -1, 12)

    # 存储为h5文件
    file = h5py.File(dir+'/data/down_Fea/DB2_s' + str(j) + 'fea.h5', 'w')
    file.create_dataset('x_train', data= train_feature.astype('float32'))
    file.create_dataset('x_test', data=test_feature.astype('float32'))

    file.create_dataset('y_train', data=y_train.astype('int32'))
    file.create_dataset('y_test', data=y_test.astype('int32'))
    file.close()

    print('******************DB2_s' + str(j) + '分割完成***********************')