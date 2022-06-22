import tensorflow as tf
import h5py
import numpy as np
import nina_funcs as nf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from Models.DBFeaEmgNet.FeaEmgFile2 import FeaAndEmg_model1, FeaAndEmg_se
from tfdeterminism import patch

#确定随机数
from Util.function import get_threeSet

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
        file = h5py.File(root_data + '/data/stimulus/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        emg, label, rep = file['emg'][:], file['label'][:], file['rep'][:]
        emg_train, emg_vali, emg_test, label_train, label_vail, label_test = get_threeSet(emg, label, rep, 6)
        file.close()
        label_train, label_vail = nf.get_categorical(label_train), nf.get_categorical(label_vail)

        feaFile = h5py.File(root_data + '/data/stimulus/Fea/DB2_s' + str(j) + 'fea.h5', 'r')
        # 将六次重复手势分开存储
        fea_all, fea_label, fea_rep = feaFile['fea_all'][:], feaFile['fea_label'][:],  feaFile['fea_rep'][:]
        feaFile.close()
        fea_train, fea_vali, fea_test, feay_train, feay_vail, feay_test = get_threeSet(fea_all, fea_label, fea_rep, 6)


        # fea_trainLabel = np.concatenate([fea_y1, fea_y3, fea_y4], axis=0)
        # fea_train = fea_train.reshape(fea_train.shape[0], fea_train.shape[1], -1)
        # fea_x6 = fea_x6.reshape(fea_x6.shape[0], fea_x6.shape[1], -1)

        callbacks = [  # 1设置学习率衰减,2保存最优模型
            # ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
            #                   cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=root_data+'/DB2_model/DB2_s' + str(j) + 'model.h5'
                            , monitor='val_accuracy', save_best_only=True)]
        model = FeaAndEmg_model1()
        # model.summary()
        history = model.fit([emg_train, fea_train], label_train, epochs=50, verbose=2, batch_size=32
                            # , callbacks=callbacks)
                            , validation_data=([emg_vali,  fea_vali],  label_vail), callbacks=callbacks)

