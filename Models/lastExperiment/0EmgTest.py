from copy import deepcopy
import h5py
from tensorflow import keras
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import nina_funcs as nf
from Models.DBEmgNet.EmgNet import EmgAway3reluBConv

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
        acc1 =(tp+tn)/(number)
        acc.append(acc1)
    return acc


if __name__ == '__main__':
    for j in range(1, 2):
        file = h5py.File(dir+'/lastdata/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        X_test = file['emg_test'][:]
        y_test = file['y_test'][:]
        file.close()
        # 选定手势做训练集，测试集，验证集

        model = keras.models.load_model(dir+'/DB2_model/DB2_s' + str(j) + 'model.h5')
        '''特征向量数据适应'''

        Y_test = nf.get_categorical(np.array(y_test))
        Y_predict = model.predict(X_test)

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

        with open("F:/DB2/result/111/DB2_s"+str(j)+"6seg.txt", "w", encoding='utf-8') as f:
            f.write(str(contexts))
            f.close()
        # print(classification_report(y_true, y_pred, target_names=classes, digits=4))


