import tensorflow as tf
import h5py
import numpy as np
import nina_funcs as nf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from Models.DBFeaEmgNet.FeaEmgFile2 import FeaAndEmg_model3, FeaAndEmg_model1
from tfdeterminism import patch

from Models.DBFeaNet.FeaModel import model1

dir='F:/DB2'

#确定随机数
patch()
np.random.seed(123)
tf.random.set_seed(123)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == '__main__':
    for j in range(1, 2):

        feaFile = h5py.File(dir + '/data/Fea/DB2_s' + str(j) + 'fea.h5', 'r')
        # 将六次重复手势分开存储
        fea_x1, fea_x3, fea_x4, fea_x6 = feaFile['fea_x1'][:], feaFile['fea_x3'][:]\
                                , feaFile['fea_x4'][:], feaFile['fea_x6'][:]
        fea_y1, fea_y3, fea_y4, fea_y6 = feaFile['fea_y1'][:], feaFile['fea_y3'][:]\
                                 , feaFile['fea_y4'][:], feaFile['fea_y6'][:]
        fea_xtest = feaFile['fea_xtest'][:]
        fea_ytest = feaFile['fea_ytest'][:]
        feaFile.close()

        fea_train = np.concatenate([fea_x1, fea_x3, fea_x4], axis=0)
        fea_trainLabel = np.concatenate([fea_y1, fea_y3, fea_y4], axis=0)

        Y_train = nf.get_categorical(fea_trainLabel)
        Y_test =nf.get_categorical(fea_y6)

        callbacks = [  # 1设置学习率衰减,2保存最优模型
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=dir+'/DB2_model/DB2_s' + str(j) + 'model.h5'
                            , monitor='val_accuracy', save_best_only=True)]
        model = model1()
        history = model.fit( fea_train, Y_train, epochs=50, verbose=2, batch_size=32
                            # , callbacks=callbacks)
                            , validation_data=(fea_y6, Y_test), callbacks=callbacks)

