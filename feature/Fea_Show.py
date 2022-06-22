from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nina_funcs as nf
from sklearn import preprocessing
import scikitplot as skplt
from sklearn.feature_selection import SelectKBest
from Util.function import get_threeSet

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
gestures = list(range(1,50))
dir='F:/DB2'


def plot_feature_scores(x, y, names=None, chname=None):
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
    ax.set_xlabel(chname+' Feature Score')
    ax.set_ylabel('Feature Name')
    ax.invert_yaxis()
    ax.set_title('F_class if scores of the features.')

    # 4. 添加每个“条条”的数字标签
    for score, pos in zip(sorted_scores, y_pos):
        ax.text(score + 20, pos, '%.1f' % score, ha='center', va='bottom', fontsize=8)

    plt.show()


root_data = 'F:/DB2'

if __name__ == '__main__':
    for j in range(1, 2):
        file = h5py.File(root_data + '/data/stimulus/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        emg, label, rep = file['emg'][:], file['label'][:], file['rep'][:]
        file.close()

        emg_train,  emg_test, label_train,  label_test = get_threeSet(emg, label, rep)

        '''step2: 选择特征组合和归一化'''
        # 选择特征组合
        # features = [nf.iemg]
        ch = 4
        features = [nf.iemg, nf.rms, nf.entropy, nf.kurtosis, nf.zero_cross, nf.min, nf.max, nf.mean, nf.median]
        temp = emg_train[:, :, ch - 1]
        temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
        train_feature = nf.feature_extractor(features=features, shape=(temp.shape[0], -1), data=temp)
        # test_feature = nf.feature_extractor(features, (x_test.shape[0], -1), x_test)
        fea_name = ['iemg', 'rms', 'entropy', 'kurtosis', 'zero_cross', 'min', 'max', 'mean', 'median']
        # fea_name = [ 'rms', 'iemg', 'min','max']
        plot_feature_scores(np.array(train_feature),  label_train, names=fea_name, chname='ch' + str(ch))
