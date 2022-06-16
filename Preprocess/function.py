import math
import h5py
import pandas as pd
import numpy as np
import tsaug
from sklearn import preprocessing
import scipy.signal as signal
import nina_funcs as nf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def get_set(emg, label, rep_arr, rep_vali):
    train_reps = [1, 3, 4, 6]
    test_reps = [2, 5]
    train_reps.remove(rep_vali)
    x = [np.where(rep_arr[:] == rep) for rep in test_reps]
    indices = np.squeeze(np.concatenate(x, axis=-1))
    emg_test = emg[indices, :]
    label_test = label[indices]
    x = [np.where(rep_arr[:] == rep_vali)]
    indices2 = np.squeeze(np.concatenate(x, axis=-1))
    emg_vali = emg[indices2, :]
    label_vail = label[indices2]
    x = [np.where(rep_arr[:] == rep) for rep in train_reps]
    indices3 = np.squeeze(np.concatenate(x, axis=-1))
    emg_train = emg[indices3, :]
    label_train = label[indices3]

    return emg_train, emg_vali, emg_test, label_train, label_vail, label_test


# if __name__ == '__main__':
#     train_reps = [1, 3, 4, 6]
#     test_reps = 3
#     train_reps.remove(test_reps)
#     train_reps
