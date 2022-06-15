import tensorflow as tf
import h5py
import numpy as np
import nina_funcs as nf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from Models.DBFeaEmgNet.FeaEmgFile import FeaAndEmg_model1
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
        file = h5py.File(dir+'/data/Comb_downSeg/DB2_s' + str(j) + 'Seg17.h5', 'r')
        emg1, emg3, emg4, emg6 = file['x_train1'][:], file['x_train3'][:], file['x_train4'][:], file['x_train6'][:]
        label1, label3, label4, label6 = file['y_train1'][:],file['y_train3'][:],file['y_train4'][:],file['y_train6'][:]
        emg_test = file['x_test'][:]
        label_test = file['y_test'][:]

        emg_train = np.concatenate([emg1, emg3, emg4, emg6], axis=0)
        Y_train = nf.get_categorical(np.concatenate([label1, label3, label4, label6], axis=0))

        Y_test = nf.get_categorical(label_test)
        file.close()


        # Xtrain1, Xtrain2, Xtrain3 = Sep3Data(X_train)
        # Xvali1, Xvali2, Xvali3 = Sep3Data(X_test)
        feaFile = h5py.File(dir+'/data/down_Fea/DB2_s' + str(j) + 'fea17.h5', 'r')
        # 将六次重复手势分开存储
        fea_train1, fea_train3, fea_train4, fea_train6 = feaFile['fea_train1'][:], feaFile['fea_train3'][:]\
                                , feaFile['fea_train4'][:], feaFile['fea_train6'][:]
        fea_label1, fea_label3, fea_label4, fea_label6 = feaFile['y_train1'][:], feaFile['y_train3'][:]\
                                 , feaFile['y_train4'][:], feaFile['y_train6'][:]
        fea_test = feaFile['fea_test'][:],
        fea_testLabel = feaFile['y_test'][:]
        feaFile.close()

        fea_train = np.concatenate([fea_train1, fea_train3, fea_train4, fea_train6], axis=0)
        fea_trainLabel = np.concatenate([fea_label1, fea_label3, fea_label4, fea_label6], axis=0)

        callbacks = [  # 1设置学习率衰减,2保存最优模型
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=dir+'/DB2_model/DB2_s' + str(j) + 'model.h5'
                            , monitor='val_accuracy', save_best_only=True)]
        model = FeaAndEmg_model1()
        history = model.fit([emg_train, fea_train], Y_train, epochs=50, verbose=2, batch_size=32
                            # , callbacks=callbacks)
                            , validation_data=([emg_test, fea_test], Y_test), callbacks=callbacks)

