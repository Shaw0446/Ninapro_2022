import tensorflow as tf
import h5py
import numpy as np
import nina_funcs as nf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from Models.PopularModel.EmgNet import EmgCNN, EmgCNN2, EmgAway3reluBConv
from Models.lastTrain.SENetFile import SENet_model
from Models.lastTrain.efficientnet import efficientnet_model1
from Models.lastTrain.modelFile import FeaAndEmg_model1, FeaAndEmgsoft_model
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



if __name__ == '__main__':
    for j in range(1, 2):
        file = h5py.File(dir+'/lastdata/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        emg1, emg3, emg4, emg6 = file['emg1'][:], file['emg3'][:], file['emg4'][:], file['emg6'][:]
        label1, label3, label4, label6 = file['y_train1'][:],file['y_train3'][:],file['y_train4'][:],file['y_train6'][:]

        X_train = np.concatenate([emg1, emg3, emg4, ], axis=0)
        y_train = np.concatenate([label1, label3, label4, ], axis=0)
        Y_train = nf.get_categorical(y_train)
        X_test = file['emg_test'][:]
        y_test = file['y_test'][:]
        Y_test = nf.get_categorical(label6)
        file.close()

        # Xtrain1, Xtrain2, Xtrain3 = Sep3Data(X_train)
        # Xvali1, Xvali2, Xvali3 = Sep3Data(X_test)
        feaFile = h5py.File(dir + '/data/Fea/DB2_s' + str(j) + 'fea.h5', 'r')
        # 将六次重复手势分开存储
        fea_train1, fea_train3, fea_train4, fea_train6 = feaFile['fea_train1'][:], feaFile['fea_train3'][:]\
                                , feaFile['fea_train4'][:], feaFile['fea_train6'][:]
        fea_label1, fea_label3, fea_label4, fea_label6 = feaFile['fea_label1'][:], feaFile['fea_label3'][:]\
                                 , feaFile['fea_label4'][:], feaFile['fea_label6'][:]
        fea_test = feaFile['fea_test'][:],
        fea_testLabel = feaFile['fea_testLabel'][:]
        feaFile.close()

        fea_train = np.concatenate([fea_train1, fea_train3, fea_train4], axis=0)
        fea_trainLabel = np.concatenate([fea_label1, fea_label3, fea_label4,], axis=0)

        callbacks = [  # 1设置学习率衰减,2保存最优模型
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=dir+'/DB2_model/DB2_s' + str(j) + 'model.h5'
                            , monitor='val_accuracy', save_best_only=True)]
        model = SENet_model()
        history = model.fit([X_train, fea_train], Y_train, epochs=50, verbose=2, batch_size=32
                            # , callbacks=callbacks)
                            , validation_data=([emg6, fea_train6], Y_test), callbacks=callbacks)

