import h5py
import pandas as pd
import numpy as np
import scipy.signal as signal
import nina_funcs as nf
from sklearn import preprocessing

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
gestures = list(range(1, 18))
dir='F:/DB2'

'''对nina_funcs继续扩展的从训练集中划分验证集'''
def dataCombin(data, label, rep_arr, reps):
    if reps:
        x = [np.where(rep_arr[:] == rep) for rep in reps]
        train_indices = np.squeeze(np.concatenate(x, axis=-1))
        train_data = data[train_indices, :]
        train_label = label[train_indices]

    return train_data, train_label


def DB_normalise(data, train_reps, channel=12):
    x = [np.where(data.values[:, channel + 1] == rep) for rep in train_reps]
    indices = np.squeeze(np.concatenate(x, axis=-1))
    train_data = data.iloc[indices, :]
    train_data = data.reset_index(drop=True)
    scaler = preprocessing.StandardScaler(with_mean=True,
                            with_std=True,
                            copy=False).fit(train_data.iloc[:, :channel])

    scaled = scaler.transform(data.iloc[:, :channel])
    normalised = pd.DataFrame(scaled)
    normalised['stimulus'] = data['stimulus'].values
    normalised['repetition'] = data['repetition'].values

    return normalised


for j in range(1, 2):
    df = pd.read_hdf(dir+'/data/filter/DB2_s' + str(j) + 'filter.h5', 'df')

    '''标准化'''
    scaler = preprocessing.StandardScaler()
    df1 = DB_normalise(df.copy(deep=True), train_reps)
    df2 = df1.copy(deep=True)
    df2.iloc[:, :12] = df2.iloc[:, :12]
    df2 = df2.astype(np.float32)
    ''' 滑动窗口分割'''
    x_train, y_train, r_train = nf.windowing(df, reps=train_reps, gestures=gestures, win_len=400, win_stride=100)
    x_test, y_test, r_test = nf.windowing(df, reps=test_reps, gestures=gestures, win_len=400, win_stride=100)



    '''数据集合的划分和组合'''
    x_train1, y_train1 = dataCombin(x_train, y_train, r_train, [1])
    x_train3, y_train3 = dataCombin(x_train, y_train, r_train, [3])
    x_train4, y_train4 = dataCombin(x_train, y_train, r_train, [4])
    x_train6, y_train6 = dataCombin(x_train, y_train, r_train, [6])



    # 存储为h5文件
    file = h5py.File(dir+'/data/Comb_Seg/DB2_s' + str(j) + 'Seg17_zsc.h5', 'w')
    file.create_dataset('x_train1', data=x_train1.astype('float32'))
    file.create_dataset('x_train3', data=x_train3.astype('float32'))
    file.create_dataset('x_train4', data=x_train4.astype('float32'))
    file.create_dataset('x_train6', data=x_train6.astype('float32'))
    file.create_dataset('y_train1', data=y_train1.astype('int'))
    file.create_dataset('y_train3', data=y_train3.astype('int'))
    file.create_dataset('y_train4', data=y_train4.astype('int'))
    file.create_dataset('y_train6', data=y_train6.astype('int'))

    file.create_dataset('x_test', data=x_test.astype('float32'))
    file.create_dataset('y_test', data=y_test.astype('int'))
    file.close()

    print('******************DB2_s' + str(j) + '分割完成***********************')
