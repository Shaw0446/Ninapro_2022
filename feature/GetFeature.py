import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nina_funcs as nf
from sklearn import preprocessing

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
# gestures = list(range(1,18))
dir='F:/DB2'

def fea_transpose(data):
    # 为了便于归一化，对矩阵进行转置
    arr_T = np.transpose(np.array(data))
    sc_fea = preprocessing.StandardScaler().fit_transform(arr_T)
    arr_fea = np.transpose(sc_fea)
    return arr_fea




for j in range(1, 2):
    file1 = h5py.File(dir+'/lastdata/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
    '''step1: 数据集划分'''
    emg1, emg3, emg4, emg6 = file1['Data1'][:], file1['Data3'][:], file1['Data4'][:], file1['Data6'][:]
    label1, label3, label4, label6 = file1['label1'][:], file1['label3'][:], file1['label4'][:], file1['label6'][:]
    emg2, emg5 = file1['Data2'][:], file1['Data5'][:]
    label2, label5 = file1['label2'][:], file1['label5'][:]

    X_emg_train = np.concatenate([emg1, emg3, emg4, emg6], axis=0)
    y_emg_train = np.concatenate([label1, label3, label4, label6], axis=0)

    x_emg_test = np.concatenate([emg2, emg5], axis=0)
    y_emg_test = np.concatenate([label2, label5], axis=0)

    '''step2: 选择特征组合和归一化'''
    # 选择特征组合
    features = [nf.rms,nf.min,nf.max]
    # features = [nf.emg_dwpt, nf.iemg,nf.rms,nf.hist,nf.entropy,nf.kurtosis,nf.zero_cross,nf.min,nf.max,nf.mean,nf.median,nf.psd]

    fea_train1 = nf.feature_extractor(features=features, shape=(emg1.shape[0], -1), data=emg1)
    fea_train3 = nf.feature_extractor(features=features, shape=(emg3.shape[0], -1), data=emg3)
    fea_train4 = nf.feature_extractor(features=features, shape=(emg4.shape[0], -1), data=emg4)
    fea_train6 = nf.feature_extractor(features=features, shape=(emg6.shape[0], -1), data=emg6)

    test_feature = nf.feature_extractor(features, (x_emg_test.shape[0], -1),  x_emg_test)

    # ss = preprocessing.Normalizer(norm="l2")
    # ss.fit(train_feature)
    # sc_train = ss.transform(train_feature)
    # sc_test = ss.transform(test_feature)

    # x_test_fea = sc_test .reshape(sc_test.shape[0], -1, 12)

    # 存储为h5文件
    file = h5py.File(dir+'/data/Fea/DB2_s' + str(j) + 'fea.h5', 'w')
    file.create_dataset('fea_train1', data=fea_train1.astype('float32'))
    file.create_dataset('fea_train3', data=fea_train3.astype('float32'))
    file.create_dataset('fea_train4', data=fea_train4.astype('float32'))
    file.create_dataset('fea_train6', data=fea_train6.astype('float32'))
    file.create_dataset('fea_label1', data=label1.astype('int'))
    file.create_dataset('fea_label3', data=label3.astype('int'))
    file.create_dataset('fea_label4', data=label4.astype('int'))
    file.create_dataset('fea_label6', data=label6.astype('int'))

    file.create_dataset('fea_test', data=test_feature.astype('float32'))
    file.create_dataset('fea_testLabel', data=y_emg_test.astype('int'))
    file.close()
    print('******************DB2_s' + str(j) + '分割完成***********************')