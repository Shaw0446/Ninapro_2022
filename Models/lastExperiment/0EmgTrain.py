import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import nina_funcs as nf

from tfdeterminism import patch

from Models.DBEmgNet.EmgFile import Away3reluBNCBAMcat, BConv
from Preprocess.function import get_set
from Util.SepData import Sep3Data

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
    root_data = 'F:/DB2'
    for j in range(1,2):
        file = h5py.File(root_data + '/data/Seg/DB2_s' + str(j) + 'Seg11.h5', 'r')
        emg, label, rep = file['emg'][:], file['label'][:], file['rep'][:]
        emg_train, emg_vali, emg_test, label_train, label_vail, label_test = get_set(emg, label, rep, 6)
        file.close()

        Y_train = nf.get_categorical(label_train)
        Y_vali = nf.get_categorical(label_vail)

        Xtrain1, Xtrain2, Xtrain3 = Sep3Data(emg_train)
        Xvali1, Xvali2, Xvali3 = Sep3Data(emg_vali)



        callbacks = [#1设置学习率衰减,2保存最优模型
            # EarlyStopping(monitor='val_accuracy', patience=5),
            # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001,
            #                   cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=root_data+'/DB2_model/DB2_s' + str(j) + 'model.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model = BConv()
        history = model.fit(emg_train, Y_train, epochs=50, verbose=2, batch_size=64
                            , validation_data=(emg_vali, Y_vali), callbacks=callbacks)


