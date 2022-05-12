import h5py
import pandas as pd
import numpy as np
import scipy.signal as signal
import nina_funcs as nf


train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
gestures = list(range(1, 50))
dir='F:/DB2'




for j in range(1, 2):
    df = pd.read_hdf(dir+'/data/filter/DB2_s' + str(j) + 'filter.h5', 'df')

    df1 = nf.normalise(df.copy(deep=True), train_reps)

    df2 = df1.copy(deep=True)
    df2.iloc[:, :12] = df2.iloc[:, :12]
    df3 = df2.astype(np.float32)
    # 滑动窗口分割
    x_train, y_train, r_train = nf.windowing(df3, reps=train_reps, gestures=gestures, win_len=400, win_stride=100)
    x_test, y_test, r_test = nf.windowing(df3, reps=test_reps, gestures=gestures, win_len=400, win_stride=100)

    # 存储为h5文件
    #     train_feature_matrix.to_hdf('F:/DB2/feature/train/DB2_s' + str(j) + 'Seg.h5', format='table', key='df', mode='w')
    #     test_feature_matrix.to_hdf('F:/DB2/feature/test/DB2_s' + str(j) + 'Seg.h5', format='table', key='df', mode='w')
    file = h5py.File(dir+'/data/NorSeg/DB2_s' + str(j) + 'Seg.h5', 'w')
    file.create_dataset('x_train', data=x_train.astype('float32'))
    file.create_dataset('x_test', data=x_test.astype('float32'))
    file.create_dataset('y_train', data=y_train.astype('int32'))
    file.create_dataset('y_test', data=y_test.astype('int32'))
    file.create_dataset('r_train', data=r_train.astype('int32'))
    file.create_dataset('r_test', data=r_test.astype('int32'))
    file.close()

    print('******************DB2_s' + str(j) + '分割完成***********************')


