import h5py
import pandas as pd

import nina_funcs as nf


train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
gestures = list(range(1, 50))
dir='F:/ninapro'



for j in range(1, 2):
    df = pd.read_hdf(dir+'/data/filter/DB2_s' + str(j) + 'filter.h5', 'df')

    '''step2: 滑动窗口分割'''
    x_train, y_train, r_train = nf.windowing(df, reps=train_reps, gestures=gestures, win_len=400, win_stride=100)
    x_test, y_test, r_test = nf.windowing(df, reps=test_reps, gestures=gestures, win_len=400, win_stride=100)

    # 存储为h5文件
    file = h5py.File(dir+'/data/Seg/DB2_s' + str(j) + 'Seg.h5', 'w')
    file.create_dataset('x_train', data=x_train.astype('float32'))
    file.create_dataset('x_test', data=x_test.astype('float32'))
    file.create_dataset('y_train', data=y_train.astype('int32'))
    file.create_dataset('y_test', data=y_test.astype('int32'))
    file.create_dataset('r_train', data=r_train.astype('int32'))
    file.create_dataset('r_test', data=r_test.astype('int32'))
    file.close()

    print('******************DB2_s' + str(j) + '分割完成***********************')


