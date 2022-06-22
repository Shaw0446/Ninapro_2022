from copy import deepcopy
import h5py
from tensorflow import keras
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import nina_funcs as nf

from Util.function import get_threeSet

dir='F:/DB2'


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(20, 16), dpi=100)
    np.set_printoptions(precision=2)  # 用于控制Python中小数的显示精度
    classes = []
    for i in range(len(cm)):
        classes.append(i)
    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (cm.size, 2))
    for i, j in iters:
        plt.text(j, i, format(cm[i, j]))  # 显示对应的数字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    # show confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.colorbar()
    plt.savefig(savename, format='png')
    plt.show()

def getACC(Y_test,Y_pred,n):
    acc =[]
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:])- tp
        fp = np.sum(con_mat[:,i])- tp
        tn = number - tp - fn - fp
        acc1 =tp/(tp+fn)
        acc.append(acc1)
    return acc

root_data = 'F:/DB2'
if __name__ == '__main__':
    for j in range(1, 2):
        file = h5py.File(root_data + '/data/stimulus/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        emg, label, rep = file['emg'][:], file['label'][:], file['rep'][:]
        emg_train, emg_vali, emg_test, label_train, label_vail, label_test = get_threeSet(emg, label, rep, 6)
        file.close()
        label_train, label_vail = nf.get_categorical(label_train), nf.get_categorical(label_vail)

        feaFile = h5py.File(root_data + '/data/stimulus/Fea/DB2_s' + str(j) + 'fea.h5', 'r')
        # 将六次重复手势分开存储
        fea_all, fea_label, fea_rep = feaFile['fea_all'][:], feaFile['fea_label'][:], feaFile['fea_rep'][:]
        feaFile.close()
        fea_train, fea_vali, fea_test, feay_train, feay_vail, feay_test = get_threeSet(fea_all, fea_label, fea_rep, 6)

        model = keras.models.load_model(dir+'/DB2_model/DB2_s' + str(j) + 'model.h5')
        '''特征向量数据适应'''
        fea_xtest = np.expand_dims(fea_test, axis=-1)

        Y_test = nf.get_categorical(np.array(label_test))
        Y_predict = model.predict([emg_test,fea_xtest])

        # # 返回每行中概率最大的元素的列坐标（热编码转为普通标签）
        y_pred = Y_predict.argmax(axis=1)
        y_true = Y_test.argmax(axis=1)
        all_class=np.array(getACC(y_true, y_pred,49))
        print("################################")
        print(getACC(y_true, y_pred,49))

        cm = confusion_matrix(y_true, y_pred)
        # plot_confusion_matrix(cm,'1C-50E-2e4.png')
        classes = []
        for i in range(len(cm)):
            classes.append(str(i))
        contexts = classification_report(y_true, y_pred, target_names=classes, digits=4)

        with open("F:/DB2/result/stimulus/111/DB2_s"+str(j)+"6seg.txt", "w", encoding='utf-8') as f:
            f.write(str(contexts))
            f.close()
        # print(classification_report(y_true, y_pred, target_names=classes, digits=4))


