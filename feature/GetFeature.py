import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nina_funcs as nf
from sklearn import preprocessing

from Util.function import get_twoSet

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
# gestures = list(range(1,18))

def fea_transpose(data):
    # 为了便于归一化，对矩阵进行转置
    arr_T = np.transpose(np.array(data))
    sc_fea = preprocessing.StandardScaler().fit_transform(arr_T)
    arr_fea = np.transpose(sc_fea)
    return arr_fea


root_data='F:/DB2'

if __name__ == '__main__':
    for j in range(1, 2):
        file = h5py.File(root_data+'/data/stimulus/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        '''step1: 数据集划分'''
        emg, label, rep = file['emg'][:], file['label'][:], file['rep'][:]
        # emg_train, emg_vali, emg_test, label_train, label_vail, label_test = get_twoSet(emg, label, rep, 6)
        file.close()

        '''step2: 选择特征组合和归一化'''
        # 选择特征组合
        features = [nf.psd]
        # features = [nf.emg_dwpt, nf.iemg,nf.rms,nf.hist,nf.entropy,nf.kurtosis,nf.zero_cross,nf.min,nf.max,nf.mean,nf.median,nf.wl]

        fea_all = nf.frequency_features_extractor(features=features, shape=(emg.shape[0], -1), data=emg)
        # fea_all = nf.feature_extractor(features=features, shape=(emg.shape[0], -1), data=emg)

        # ss = preprocessing.Normalizer(norm="l2")
        # ss.fit(train_feature)
        # sc_train = ss.transform(train_feature)
        # sc_test = ss.transform(test_feature)

        # x_test_fea = sc_test .reshape(sc_test.shape[0], -1, 12)

        # 存储为h5文件
        file = h5py.File(root_data+'/data/stimulus/Fea/DB2_s' + str(j) + 'fea.h5', 'w')
        file.create_dataset('fea_all', data=fea_all.astype('float32'))
        file.create_dataset('fea_label', data=label.astype('int'))
        file.create_dataset('fea_rep', data=rep.astype('int'))


        file.close()
        print('******************DB2_s' + str(j) + '分割完成***********************')