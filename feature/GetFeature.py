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



if __name__ == '__main__':
    for j in range(1, 2):
        file = h5py.File(dir+'/lastdata/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        '''step1: 数据集划分'''
        emg1, emg3, emg4, emg6 = file['emg1'][:], file['emg3'][:], file['emg4'][:], file['emg6'][:]
        y1, y3, y4, y6 = file['y1'][:], file['y3'][:], file['y4'][:], file['y6'][:]


        # x_emg_train = np.concatenate([emg1, emg3, emg4, emg6], axis=0)
        # y_emg_train = np.concatenate([y1, y3, y4, y6], axis=0)

        emg_test = file['emg_test'][:]
        y_test = file['y_test'][:]

        '''step2: 选择特征组合和归一化'''
        # 选择特征组合
        features = [nf.rms,nf.min,nf.max]
        # features = [nf.emg_dwpt, nf.iemg,nf.rms,nf.hist,nf.entropy,nf.kurtosis,nf.zero_cross,nf.min,nf.max,nf.mean,nf.median,nf.psd]

        fea_x1 = nf.feature_extractor(features=features, shape=(emg1.shape[0], -1), data=emg1)
        fea_x3 = nf.feature_extractor(features=features, shape=(emg3.shape[0], -1), data=emg3)
        fea_x4 = nf.feature_extractor(features=features, shape=(emg4.shape[0], -1), data=emg4)
        fea_x6 = nf.feature_extractor(features=features, shape=(emg6.shape[0], -1), data=emg6)

        fea_test = nf.feature_extractor(features, (emg_test.shape[0], -1), emg_test)

        # ss = preprocessing.Normalizer(norm="l2")
        # ss.fit(train_feature)
        # sc_train = ss.transform(train_feature)
        # sc_test = ss.transform(test_feature)

        # x_test_fea = sc_test .reshape(sc_test.shape[0], -1, 12)

        # 存储为h5文件
        file = h5py.File(dir+'/data/Fea/DB2_s' + str(j) + 'fea.h5', 'w')
        file.create_dataset('fea_x1', data=fea_x1.astype('float32'))
        file.create_dataset('fea_x3', data=fea_x3.astype('float32'))
        file.create_dataset('fea_x4', data=fea_x4.astype('float32'))
        file.create_dataset('fea_x6', data=fea_x6.astype('float32'))
        file.create_dataset('fea_y1', data=y1.astype('int'))
        file.create_dataset('fea_y3', data=y3.astype('int'))
        file.create_dataset('fea_y4', data=y4.astype('int'))
        file.create_dataset('fea_y6', data=y6.astype('int'))

        file.create_dataset('fea_xtest', data=fea_test.astype('float32'))
        file.create_dataset('fea_ytest', data=y_test.astype('int'))
        file.close()
        print('******************DB2_s' + str(j) + '分割完成***********************')