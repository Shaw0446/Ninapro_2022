import h5py
import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import nina_funcs as nf

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
gestures = list(range(1,50))
dir='F:/DB2'


for j in range(1, 2):
    file = h5py.File(dir+'/data/down_Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
    '''step1: 数据集划分'''
    x_train, y_train, r_train = file['x_train'][:], file['y_train'][:], file['r_train'][:]
    x_test, y_test, r_test = file['x_test'][:], file['y_test'][:], file['r_test'][:]
    file.close()

    '''step2: 选择特征组合'''
    #选择特征组合
    features = [nf.emg_dwpt]
    # features = [nf.rms, nf.entropy,  nf.kurtosis, nf.zero_cross, nf.min, nf.max, nf.mean, nf.median]
    train_feature_matrix = nf.feature_extractor(features=features, shape=(x_train.shape[0], -1), data=x_train[:50,:])
    arr_train = np.array(train_feature_matrix)
    # X_train_fea = arr_train.reshape(X_train.shape[0], -1, 12)

    test_feature_matrix = nf.feature_extractor(features, (x_test.shape[0], -1), x_test)
    arr_test = np.array(test_feature_matrix)

    # 存储为h5文件
    file = h5py.File(dir+'/data/down_Fea/DB2_s' + str(j) + 'fea.h5', 'w')
    file.create_dataset('x_train', data=arr_train.astype('float32'))
    file.create_dataset('x_test', data=arr_test.astype('float32'))

    file.create_dataset('y_train', data=y_train.astype('int32'))
    file.create_dataset('y_test', data=y_test.astype('int32'))
    file.close()

    print('******************DB2_s' + str(j) + '分割完成***********************')