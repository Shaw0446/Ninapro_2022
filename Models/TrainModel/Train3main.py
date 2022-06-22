import tensorflow as tf
import h5py
import numpy as np
import nina_funcs as nf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from Models.DBEmgNet.CBAMFile import Away3reluBNCBAMcat
from Util.SepData import Sep3Data
from tfdeterminism import patch

from Util.function import get_set

root_data='F:/DB2'

#确定随机数
os.environ['TF_DETERMINISTIC_OPS'] = '1'
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

if __name__ == '__main__':
    for j in range(1, 2):
        file = h5py.File(root_data + '/data/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        # 数据集划分呢
        file = h5py.File(root_data + '/data/stimulus/Seg/DB2_s' + str(j) + 'Seg11.h5', 'r')
        emg, label, rep = file['emg'][:], file['label'][:], file['rep'][:]
        emg_train, emg_vali, emg_test, label_train, label_vail, label_test = get_set(emg, label, rep, 6)
        file.close()

        emg_train, emg_vali, emg_test, label_train, label_vail, label_test = get_set(emg, label, rep, 4)

        Xtrain1, Xtrain2, Xtrain3 = Sep3Data(emg_train)
        Xvali1, Xvali2, Xvali3 = Sep3Data(emg_vali)

        Y_train = nf.get_categorical(label_train)
        Y_vali= nf.get_categorical(label_test)

        callbacks = [#1设置学习率衰减,2保存最优模型
            # ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
            #                   cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=root_data+'/DB2_model/DB2_s' + str(j) + 'seg.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model = Away3reluBNCBAMcat()
        history = model.fit([Xtrain1, Xtrain2, Xtrain3], Y_train, epochs=50, verbose=2, batch_size=64
            ,validation_data=([Xvali1, Xvali2, Xvali3],Y_vali), callbacks=callbacks)

        # loss= history.history['loss']
        # val_loss = history.history['val_loss']
        # accuracy =history.history['accuracy']
        # val_accuracy =history.history['val_accuracy']32
        # pltCurve(loss, val_loss, accuracy, val_accuracy)
        #早停机制保存最优模型后，不再另外保存
        # model.save('D:/Pengxiangdong/ZX/modelsave/3Channel/Away3CBAMcat/DB2_s'+str(j)+'re25seg400100mZsc.h5')
        # tf.keras.utils.plot_model(model, to_file='../ModelPng/Away3reluBNCBAMcatNEW.png', show_shapes=True)

