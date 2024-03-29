import h5py
import pandas as pd
import numpy as np
import scipy.signal as signal
import nina_funcs as nf
from sklearn import preprocessing

from Preprocess.Combination import dataCombin

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
gestures = list(range(1, 18))
dir='F:/DB2'



def DB_normalise(data, train_reps, channel=12):
    x = [np.where(data.values[:, channel + 1] == rep) for rep in train_reps]
    indices = np.squeeze(np.concatenate(x, axis=-1))
    train_data = data.iloc[indices, :]
    # reset_index对行号重新排序
    train_data = data.reset_index(drop=True)
    # scaler = preprocessing.StandardScaler(with_mean=True,
    #                         with_std=True,
    #                         copy=False).fit(train_data.iloc[:, :channel])
    scaler = preprocessing.MinMaxScaler().fit(train_data.iloc[:, :channel])
    scaled = scaler.transform(data.iloc[:, :channel])
    normalised = pd.DataFrame(scaled)
    normalised['stimulus'] = data['stimulus'].values
    normalised['repetition'] = data['repetition'].values

    return normalised

if __name__ == '__main__':
    for j in range(1, 2):
        df = pd.read_hdf(dir+'/data/down/DB2_s' + str(j) + 'down.h5', 'df')

        '''标准化'''
        df1 = DB_normalise(df.copy(deep=True), train_reps)

        ''' 滑动窗口分割'''
        x_train, y_train, r_train = nf.windowing(df1, reps=train_reps, gestures=gestures, win_len=20, win_stride=1)
        x_test, y_test, r_test = nf.windowing(df1, reps=test_reps, gestures=gestures, win_len=20, win_stride=1)



        '''数据集合的划分和组合'''
        x_train1, y_train1, r_train1 = dataCombin(x_train, y_train, r_train, [1])
        x_train3, y_train3, r_train3 = dataCombin(x_train, y_train, r_train, [3])
        x_train4, y_train4, r_train4 = dataCombin(x_train, y_train, r_train, [4])
        x_train6, y_train6, r_train6 = dataCombin(x_train, y_train, r_train, [6])


        # 存储为h5文件
        file = h5py.File(dir+'/data/Comb_downSeg/DB2_s' + str(j) + 'Seg17.h5', 'w')
        file.create_dataset('x_train1', data=x_train1.astype('float32'))
        file.create_dataset('x_train3', data=x_train3.astype('float32'))
        file.create_dataset('x_train4', data=x_train4.astype('float32'))
        file.create_dataset('x_train6', data=x_train6.astype('float32'))
        file.create_dataset('y_train1', data=y_train1.astype('int'))
        file.create_dataset('y_train3', data=y_train3.astype('int'))
        file.create_dataset('y_train4', data=y_train4.astype('int'))
        file.create_dataset('y_train6', data=y_train6.astype('int'))
        file.create_dataset('r_train1', data=r_train1.astype('int'))
        file.create_dataset('r_train3', data=r_train3.astype('int'))
        file.create_dataset('r_train4', data=r_train4.astype('int'))
        file.create_dataset('r_train6', data=r_train6.astype('int'))

        file.create_dataset('x_test', data=x_test.astype('float32'))
        file.create_dataset('y_test', data=y_test.astype('int'))
        file.create_dataset('r_test', data=r_test.astype('int'))
        file.close()

        print('******************DB2_s' + str(j) + '分割完成***********************')
