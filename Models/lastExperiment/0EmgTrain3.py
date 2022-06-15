import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import nina_funcs as nf

from Models.DBEmgNet.CBAMFile import reluBNCBAMcat
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
        file = h5py.File(dir + '/lastdata/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        emg1, emg3, emg4, emg6 = file['emg1'][:], file['emg3'][:], file['emg4'][:], file['emg6'][:]
        y1, y3, y4, y6 = file['y1'][:], file['y3'][:], file['y4'][:], file['y6'][:]
        X_train = np.concatenate([emg1, emg3, emg4, ], axis=0)
        y_train = np.concatenate([y1, y3, y4,  ], axis=0)
        Y_train = nf.get_categorical(y_train)
        # X_test = file['emg_test'][:]
        # y_vali = file['y_test'][:]
        Y_vale = nf.get_categorical(y6)
        file.close()


        callbacks = [#1设置学习率衰减,2保存最优模型
            # EarlyStopping(monitor='val_accuracy', patience=5),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=dir+'/DB2_model/DB2_s' + str(j) + 'Fea.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model =  reluBNCBAMcat()
        history = model.fit(X_train, Y_train, epochs=100, verbose=2, batch_size=32
                            , validation_data=(emg6, Y_vale), callbacks=callbacks)


