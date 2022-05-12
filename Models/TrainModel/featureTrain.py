import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import nina_funcs as nf
from Models.PopularModel.FeaModel import FeaAway3CBAM, Stage1, Stage2
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
        sigFile = h5py.File(dir+'/data/down_Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        feaFile = h5py.File(dir+'/data/featureMap/DB2_s' + str(j) + 'map.h5', 'r')
        #将六次重复手势分开存储

        x_train = sigFile['x_train'][:]
        x_test = sigFile['x_test'][:]

        X_train_highimg, X_train_lowimg = feaFile['X_train_highimg'][:], feaFile['X_train_lowimg'][:]
        X_test_highimg, X_test_lowimg = feaFile['X_test_highimg'][:], feaFile['X_test_lowimg'][:]
        y_train, y_test = sigFile['y_train'][:], sigFile['y_test'][:]
        sigFile.close(), feaFile.close()
        #
        # Xtrain1, Xtrain2, Xtrain3 = Sep3Fea(X_train_highimg)
        # Xtest1, Xtest2, Xtest3 = Sep3Fea(X_test_highimg)

        Y_train = nf.get_categorical(y_train)
        Y_test = nf.get_categorical(y_test)

        callbacks = [#1设置学习率衰减,2保存最优模型
            # EarlyStopping(monitor='val_accuracy', patience=5),
            # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
            #                   cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=dir+'/DB2_model/DB2_s' + str(j) + 'Fea.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model = Stage1()
        history = model.fit([x_train ,X_train_highimg, X_train_lowimg], Y_train, epochs=50, verbose=2, batch_size=64
                            , validation_data=([x_test, X_test_highimg, X_test_lowimg], Y_test), callbacks=callbacks)

        # loss= history.history['loss']
        # val_loss = history.history['val_loss']
        # accuracy =history.history['accuracy']
        # val_accuracy =history.history['val_accuracy']32
        # pltCurve(loss, val_loss, accuracy, val_accuracy)
        #早停机制保存最优模型后，不再另外保存
        # model.save('D:/Pengxiangdong/ZX/modelsave/3Channel/Away3CBAMcat/DB2_s'+str(j)+'re25seg400100mZsc.h5')
        # tf.keras.utils.plot_model(model, to_file='../ModelPng/Away3reluBNCBAMcatNEW.png', show_shapes=True)

