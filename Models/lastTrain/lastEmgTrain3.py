import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import nina_funcs as nf

from Models.PopularModel.FeaAndEmg import FeaAndEmg_model1
from Models.PopularModel.FeaModel import FeaAway3CBAM, Stage1, Stage2, model1
from Util.SepData import Sep3Data, Sep3Fea
from tfdeterminism import patch
dir='F:/DB2'

#确定随机数
patch()
np.random.seed(123)
tf.random.set_seed(123)



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

if __name__ == '__main__':
    for j in range(1,2):
        file = h5py.File(dir + '/lastdata/Seg/DB2_s' + str(j) + 'Seg17.h5', 'r')
        emg1, emg3, emg4, emg6 = file['Data1'][:], file['Data3'][:], file['Data4'][:], file['Data6'][:]
        label1, label3, label4, label6 = file['label1'][:], file['label3'][:], file['label4'][:], file['label6'][:]
        emg2, emg5 = file['Data2'][:], file['Data5'][:]
        label2, label5 = file['label2'][:], file['label5'][:]

        X_train = np.concatenate([emg1, emg3, emg4, emg6], axis=0)
        y_train = np.concatenate([label1, label3, label4, label6], axis=0)
        Y_train = nf.get_categorical(y_train)

        X_test = np.concatenate([emg2, emg5], axis=0)
        y_test = np.concatenate([label2, label5], axis=0)
        Y_test = nf.get_categorical(y_test)
        file.close()


        callbacks = [#1设置学习率衰减,2保存最优模型
            # EarlyStopping(monitor='val_accuracy', patience=5),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=dir+'/DB2_model/DB2_s' + str(j) + 'Fea.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model = FeaAndEmg_model1()
        history = model.fit(X_train, Y_train, epochs=100, verbose=2, batch_size=32
                            , validation_data=(X_test, Y_test), callbacks=callbacks)


