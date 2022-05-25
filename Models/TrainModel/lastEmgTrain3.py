import tensorflow as tf
import h5py
import numpy as np
import nina_funcs as nf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from Models.PopularModel.Away3CBAMNEW import Away3reluBNCBAMcatNEW
from Models.PopularModel.DownAway3CBAM import DownAway3reluBNCBAM
from Models.PopularModel.EmgNet import EmgCNN, EmgCNN2, EmgAway3reluBConv
from Util.SepData import Sep3Data
from tfdeterminism import patch
dir='F:/DB2'

#确定随机数
patch()
np.random.seed(123)
tf.random.set_seed(123)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def pltCurve(loss, val_loss, accuracy, val_accuracy):
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.plot(epochs, accuracy, label='Training accuracy')
    plt.plot(epochs, val_accuracy, label='Validation val_accuracy')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def restore(array):
    N1 = 4;
    N2 = 100
    X = np.zeros([array.shape[0], N1, N2, array.shape[2]])
    for i in range(len(array)):
        temp = array[i, :, :]
        for j in range(12):
            temp2 = temp[:,j].reshape(N1, N2)
            X[i, :, :, j] = temp2

    return X





if __name__ == '__main__':
    for j in range(1, 2):
        file = h5py.File(dir+'/lastdata/Seg/DB2_s' + str(j) + 'Seg17.h5', 'r')
        emg1, emg3, emg4, emg6 = file['Data1'][:], file['Data3'][:], file['Data4'][:], file['Data6'][:]
        label1, label3, label4,label6 = file['label1'][:], file['label3'][:], file['label4'][:], file['label6'][:]
        emg2, emg5 = file['Data2'][:], file['Data5'][:]
        label2,label5=file['label2'][:], file['label5'][:]


        X_train = np.concatenate([emg1, emg3, emg4, emg6], axis=0)
        y_train = np.concatenate([label1, label3, label4, label6], axis=0)
        Y_train = nf.get_categorical(y_train)
        X_test = np.concatenate([emg2, emg5], axis=0)
        y_test = np.concatenate([label2, label5], axis=0)
        Y_test = nf.get_categorical(y_test)
        file.close()

        Xtrain1, Xtrain2, Xtrain3 = Sep3Data(X_train)
        Xvali1, Xvali2, Xvali3 = Sep3Data(X_test)


        callbacks = [  # 1设置学习率衰减,2保存最优模型
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=dir+'/DB2_model/DB2_s' + str(j) + 'model.h5'
                            , monitor='val_accuracy', save_best_only=True)]
        model = Away3reluBNCBAMcatNEW()
        history = model.fit([Xtrain1, Xtrain2, Xtrain3], Y_train, epochs=50, verbose=2, batch_size=32
                            # , callbacks=callbacks)
                            , validation_data=([Xvali1, Xvali2, Xvali3], Y_test), callbacks=callbacks)

