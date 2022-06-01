import os
import numpy as np
import glob
import scipy.io as scio

dir='F:/ninapro/DB2/'
def process_dataset_for_oneSubject(_subject):
    aaa= glob.glob( dir+_subject+'/*.mat')
    data_paths = sorted(glob.glob( dir+_subject+'/*.mat'))

    EMGData1 = scio.loadmat(data_paths[0])
    EMGData2 = scio.loadmat(data_paths[1])
    EMGData3 = scio.loadmat(data_paths[2])
    emg1 = EMGData1['emg']
    restimulus1 = EMGData1['restimulus']
    rerepetition1 = EMGData1['rerepetition']
    emg2 = EMGData2['emg']
    restimulus2 = EMGData2['restimulus']
    restimulus2 = restimulus2 + restimulus1.max() * (restimulus2>0).astype('int')
    rerepetition2 = EMGData2['rerepetition']
    emg3 = EMGData3['emg']
    restimulus3 = EMGData3['restimulus']
    restimulus3 = restimulus3 + restimulus2.max() * (restimulus3>0).astype('int')
    rerepetition3 = EMGData3['rerepetition']
    emg = np.vstack([emg1,emg2,emg3])
    restimulus = np.vstack([restimulus1,restimulus2,restimulus3]).astype('int')
    rerepetition = np.vstack([rerepetition1,rerepetition2,rerepetition3]).astype('int')
    if not os.path.exists('../data/ninapro/db1_processed/{}'.format(_subject)):
        os.makedirs('../data/ninapro/db1_processed/{}'.format(_subject))
    np.savetxt('../data/ninapro/db1_processed/{}/emg.txt'.format(_subject), emg)
    np.savetxt('../data/ninapro/db1_processed/{}/restimulus.txt'.format(_subject), restimulus, fmt="%d")
    np.savetxt('../data/ninapro/db1_processed/{}/rerepetition.txt'.format(_subject), rerepetition, fmt="%d")

if __name__ == '__main__':
    _subjects = sorted(os.listdir(dir+'/'))
    _subjects = [_subject for _subject in _subjects if not _subject.endswith('zip')]
    for _subject in _subjects:
        process_dataset_for_oneSubject(_subject)