import tensorflow as tf
import h5py
import numpy as np
import nina_funcs as nf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from Models.DBFeaEmgNet.FeaEmgFile2 import FeaAndEmg_model1, FeaAndEmg_se
from tfdeterminism import patch

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
    root_data = 'F:/DB2'
    for j in range(1, 2):
        file = h5py.File(root_data+'/lastdata/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        emg1, emg3, emg4, emg6 = file['emg1'][:], file['emg3'][:], file['emg4'][:], file['emg6'][:]
        y1, y3, y4, y6 = file['y1'][:], file['y3'][:], file['y4'][:], file['y6'][:]

        X_train = np.concatenate([emg1, emg3, emg4, ], axis=0)
        y_train = np.concatenate([y1, y3, y4, ], axis=0)
        Y_train = nf.get_categorical(y_train)
        X_test = file['emg_test'][:]
        y_test = file['y_test'][:]
        Y_test = nf.get_categorical(y6)
        file.close()

        # Xtrain1, Xtrain2, Xtrain3 = Sep3Data(X_train)
        # Xvali1, Xvali2, Xvali3 = Sep3Data(X_test)
        feaFile = h5py.File(root_data + '/data/Fea/DB2_s' + str(j) + 'fea.h5', 'r')
        # 将六次重复手势分开存储
        fea_x1, fea_x3, fea_x4, fea_x6 = feaFile['fea_x1'][:], feaFile['fea_x3'][:]\
                                , feaFile['fea_x4'][:], feaFile['fea_x6'][:]
        fea_y1, fea_y3, fea_y4, fea_y6 = feaFile['fea_y1'][:], feaFile['fea_y3'][:]\
                                 , feaFile['fea_y4'][:], feaFile['fea_y6'][:]
        fea_xtest = feaFile['fea_xtest'][:]
        fea_ytest = feaFile['fea_ytest'][:]
        feaFile.close()

        fea_train = np.concatenate([fea_x1, fea_x3, fea_x4], axis=0)

        # fea_trainLabel = np.concatenate([fea_y1, fea_y3, fea_y4], axis=0)
        # fea_train = fea_train.reshape(fea_train.shape[0], fea_train.shape[1], -1)
        # fea_x6 = fea_x6.reshape(fea_x6.shape[0], fea_x6.shape[1], -1)

        callbacks = [  # 1设置学习率衰减,2保存最优模型
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=dir+'/DB2_model/DB2_s' + str(j) + 'model.h5'
                            , monitor='val_accuracy', save_best_only=True)]
        model = FeaAndEmg_se()
        model.summary()
        history = model.fit([X_train, fea_train], Y_train, epochs=50, verbose=2, batch_size=32
                            # , callbacks=callbacks)
                            , validation_data=([emg6, fea_x6], Y_test), callbacks=callbacks)

