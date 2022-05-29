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
        file1 = h5py.File(dir+'/lastdata/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        emg1, emg3, emg4, emg6 = file1['Data1'][:], file1['Data3'][:], file1['Data4'][:], file1['Data6'][:]
        label1, label3, label4, label6 = file1['label1'][:], file1['label3'][:], file1['label4'][:], file1['label6'][:]
        emg2, emg5 = file1['Data2'][:], file1['Data5'][:]
        label2, label5 = file1['label2'][:], file1['label5'][:]

        X_train = np.concatenate([emg1, emg3, emg4, emg6], axis=0)
        y_train = np.concatenate([label1, label3, label4, label6], axis=0)
        Y_train = nf.get_categorical(y_train)
        X_test = np.concatenate([emg2, emg5], axis=0)
        y_test = np.concatenate([label2, label5], axis=0)
        Y_test = nf.get_categorical(y_test)
        file1.close()

        feaFile = h5py.File(dir+'/data/Fea/DB2_s' + str(j) + 'fea.h5', 'r')
        # 将六次重复手势分开存储
        fea_train1, fea_train3,fea_train4,fea_train6 = feaFile['fea_train1'][:], feaFile['fea_train3'][:],feaFile['fea_train4'][:], feaFile['fea_train6'][:]
        fea_label1, fea_label3,fea_label4,fea_label6 = feaFile['fea_label1'][:], feaFile['fea_label3'][:],feaFile['fea_label4'][:], feaFile['fea_label6'][:]
        fea_test = feaFile['fea_test'][:],
        fea_testLabel = feaFile['fea_testLabel'][:]
        feaFile.close()

        fea_train = np.concatenate([fea_train1, fea_train3,fea_train4,fea_train6], axis=0)
        fea_trainLabel = np.concatenate([fea_label1, fea_label3, fea_label4, fea_label6], axis=0)


        callbacks = [#1设置学习率衰减,2保存最优模型
            # EarlyStopping(monitor='val_accuracy', patience=5),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=dir+'/DB2_model/DB2_s' + str(j) + 'Fea.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model = FeaAndEmg_model1()
        history = model.fit([X_train, fea_train], Y_train, epochs=100, verbose=2, batch_size=32
                            , validation_data=([X_test,fea_test ], Y_test), callbacks=callbacks)

        # loss= history.history['loss']
        # val_loss = history.history['val_loss']
        # accuracy =history.history['accuracy']
        # val_accuracy =history.history['val_accuracy']32
        # pltCurve(loss, val_loss, accuracy, val_accuracy)
        #早停机制保存最优模型后，不再另外保存
        # model.save('D:/Pengxiangdong/ZX/modelsave/3Channel/Away3CBAMcat/DB2_s'+str(j)+'re25seg400100mZsc.h5')
        # tf.keras.utils.plot_model(model, to_file='../ModelPng/Away3reluBNCBAMcatNEW.png', show_shapes=True)

