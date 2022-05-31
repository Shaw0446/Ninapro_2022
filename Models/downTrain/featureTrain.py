import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import nina_funcs as nf
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
        File = h5py.File(dir+'/data/down_Fea/DB2_s' + str(j) + 'fea17.h5', 'r')
        #将六次重复手势分开存储
        f1, f3, f4, f6 = File['fea_train1'][:], File['fea_train3'][:], File['fea_train4'][:], File['fea_train6'][:]
        y1, y3, y4, y6 = File['y_train1'][:], File['y_train3'][:], File['y_train4'][:], File['y_train6'][:]
        x_train = np.concatenate([f1, f3, f4, f6], axis=0)
        y_train = np.concatenate([y1, y3, y4, y6], axis=0)
        x_test, y_test = File['fea_test'][:], File['y_test'][:]

        File.close()
        Y_train = nf.get_categorical(y_train)
        Y_test = nf.get_categorical(y_test)

        callbacks = [#1设置学习率衰减,2保存最优模型
            # EarlyStopping(monitor='val_accuracy', patience=5),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=dir+'/DB2_model/DB2_s' + str(j) + 'Fea.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model = model1()
        history = model.fit(x_train, Y_train, epochs=100, verbose=2, batch_size=32
                            , validation_data=(x_test, Y_test), callbacks=callbacks)

        # loss= history.history['loss']
        # val_loss = history.history['val_loss']
        # accuracy =history.history['accuracy']
        # val_accuracy =history.history['val_accuracy']32
        # pltCurve(loss, val_loss, accuracy, val_accuracy)
        #早停机制保存最优模型后，不再另外保存
        # model.save('D:/Pengxiangdong/ZX/modelsave/3Channel/Away3CBAMcat/DB2_s'+str(j)+'re25seg400100mZsc.h5')
        # tf.keras.utils.plot_model(model, to_file='../ModelPng/Away3reluBNCBAMcatNEW.png', show_shapes=True)

