import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import nina_funcs as nf

from Models.DBFeaEmgNet.FeaEmgFile import FeaAndEmg_model1
from tfdeterminism import patch
root_data='F:/DB2'

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
        file = h5py.File(root_data + '/data/Comb_Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        emg1, emg3, emg4, emg6 = file['x_train1'][:], file['x_train3'][:], file['x_train4'][:], file['x_train6'][:]
        label1, label3, label4, label6 = file['y_train1'][:],file['y_train3'][:],file['y_train4'][:],file['y_train6'][:]


        X_train = np.concatenate([emg1, emg3, emg4, emg6], axis=0)
        y_train = np.concatenate([label1, label3, label4, label6], axis=0)
        Y_train = nf.get_categorical(y_train)
        X_test = file['x_test'][:]
        y_test = file['y_test'][:]
        Y_test = nf.get_categorical(y_test)
        file.close()

        feaFile = h5py.File(root_data + '/data/Fea/DB2_s' + str(j) + 'fea.h5', 'r')
        # 将六次重复手势分开存储
        fea_x1, fea_x3, fea_x4, fea_x6 = feaFile['fea_x1'][:], feaFile['fea_x3'][:] \
            , feaFile['fea_x4'][:], feaFile['fea_x6'][:]
        fea_y1, fea_y3, fea_y4, fea_y6 = feaFile['fea_y1'][:], feaFile['fea_y3'][:] \
            , feaFile['fea_y4'][:], feaFile['fea_y6'][:]
        fea_xtest = feaFile['fea_xtest'][:]
        fea_ytest = feaFile['fea_ytest'][:]
        feaFile.close()

        fea_train = np.concatenate([fea_x1, fea_x3, fea_x4], axis=0)
        fea_trainLabel = np.concatenate([fea_y1, fea_y3, fea_y4], axis=0)

        callbacks = [#1设置学习率衰减,2保存最优模型
            # EarlyStopping(monitor='val_accuracy', patience=5),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=root_data+'/DB2_model/DB2_s' + str(j) + 'Fea.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model = FeaAndEmg_model1()
        history = model.fit([X_train, fea_train], Y_train, epochs=100, verbose=2, batch_size=32
                            , validation_data=([X_test,fea_xtest], Y_test), callbacks=callbacks)

        # loss= history.history['loss']
        # val_loss = history.history['val_loss']
        # accuracy =history.history['accuracy']
        # val_accuracy =history.history['val_accuracy']32
        # pltCurve(loss, val_loss, accuracy, val_accuracy)
        #早停机制保存最优模型后，不再另外保存
        # model.save('D:/Pengxiangdong/ZX/modelsave/3Channel/Away3CBAMcat/DB2_s'+str(j)+'re25seg400100mZsc.h5')
        # tf.keras.utils.plot_model(model, to_file='../ModelPng/Away3reluBNCBAMcatNEW.png', show_shapes=True)

