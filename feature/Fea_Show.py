from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nina_funcs as nf
from sklearn import preprocessing
import scikitplot as skplt
from sklearn.feature_selection import SelectKBest

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
gestures = list(range(1,50))
dir='F:/DB2'


def plot_feature_scores(x, y, names=None):
    if not names:
        names = range(len(x[0]))

    # 1. 使用 sklearn.feature_selection.SelectKBest 给特征打分
    slct = SelectKBest(k="all")
    slct.fit(x, y)
    scores = slct.scores_

    # 2. 将特征按分数 从大到小 排序
    named_scores = zip(names, scores)
    sorted_named_scores = sorted(named_scores, key=lambda z: z[1], reverse=True)

    sorted_scores = [each[1] for each in sorted_named_scores]
    sorted_names = [each[0] for each in sorted_named_scores]

    y_pos = np.arange(len(names))  # 从上而下的绘图顺序

    # 3. 绘图
    fig, ax = plt.subplots()
    ax.barh(y_pos, sorted_scores, height=0.7, align='center', color='#AAAAAA', tick_label=sorted_names)
    # ax.set_yticklabels(sorted_names)      # 也可以在这里设置“条条”的标签~
    ax.set_yticks(y_pos)
    ax.set_xlabel('Feature Score')
    ax.set_ylabel('Feature Name')
    ax.invert_yaxis()
    ax.set_title('F_classif scores of the features.')

    # 4. 添加每个“条条”的数字标签
    for score, pos in zip(sorted_scores, y_pos):
        ax.text(score + 20, pos, '%.1f' % score, ha='center', va='bottom', fontsize=8)

    plt.show()



for j in range(1, 2):
    file = h5py.File(dir+'/data/Comb_downSeg/DB2_s' + str(j) + 'Seg17.h5', 'r')
    '''step1: 数据集划分'''
    x_train1, x_train3, x_train4, x_train6 = file['x_train1'][:], file['x_train3'][:], file['x_train4'][:], file['x_train6'][:]
    y_train1, y_train3, y_train4, y_train6 = file['y_train1'][:], file['y_train3'][:], file['y_train4'][:], file['y_train6'][:]
    x_test = file['x_test'][:]
    y_test = file['y_test'][:]
    file.close()

    x_train = np.concatenate([x_train1, x_train3, x_train4, x_train6], axis=0)
    y_train = np.concatenate([y_train1, y_train3, y_train4, y_train6], axis=0)

    '''step2: 选择特征组合和归一化'''
    # 选择特征组合
    features = [nf.mean,nf.median,nf.psd]
    # features = [nf.emg_dwpt, nf.iemg,nf.rms,nf.hist,nf.entropy,nf.kurtosis,nf.zero_cross,nf.min,nf.max,nf.mean,nf.median,nf.psd]
    temp = x_train[:, :, :1]
    train_feature = nf.feature_extractor(features=features, shape=(temp.shape[0], -1), data=temp)
    # test_feature = nf.feature_extractor(features, (x_test.shape[0], -1), x_test)
    aaa = x_train[:, :, 0]
    fea_name=['emg_dwpt', 'iemg','rms','hist','entropy','kurtosis','zero_cross','min','max','mean','median','fft','psd']
    plot_feature_scores(np.array(train_feature),y_train,)