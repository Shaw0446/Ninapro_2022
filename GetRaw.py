import h5py
import numpy as np
import nina_funcs as nf
import pandas as pd
import sklearn

root_data='F:/ninapro/DB4'

if __name__ == '__main__':
    for j in range(1, 2):
        df1 = nf.get_redata(root_data + '/s' + str(j), 'S' + str(j) + '_E1_A1.mat')
        df2 = nf.get_redata(root_data + '/s' + str(j), 'S' + str(j) + '_E2_A1.mat')
        df2.loc[df2[df2['restimulus'] != 0].index.tolist(), 'restimulus'] += +12
        # data.loc[data[data.id == 5].index.tolist(),‘value’] = 10
        df3 = nf.get_redata(root_data + '/s' + str(j), 'S' + str(j) + '_E3_A1.mat')
        df3.loc[df3[df3['restimulus'] != 0].index.tolist(), 'restimulus'] += +29

        dfall = pd.concat([df1, df2, df3], ignore_index=True)
        # 考虑数据放大到1
        dfmax = dfall.copy(deep=True)
        dfmax.iloc[:, :12] = dfmax.iloc[:, :12]
        df = dfmax.astype(np.float32)
        df.to_hdf( 'F:/DB4/data/restimulus/raw/DB4_s' + str(j) + 'raw.h5', format='table', key='df', mode='w',
                  complevel=9, complib='blosc')

        print('******************DB4_s' + str(j) + '读取完成***********************')

        # file = h5py.File('F:/DB2/raw/DB2_s' + str(j) + 'raw.h5', 'w')
        # file.create_dataset('alldata', data=(dfall).astype('float32'))
        # file.close()











