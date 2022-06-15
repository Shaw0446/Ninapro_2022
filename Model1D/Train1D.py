import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from tfdeterminism import patch

#确定随机数
from Model1D.Away12CNN1D import DownAway12reluBNCNN1D
from Util.SepData import Sep12Data

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

    for j in range(1,3):

        file = h5py.File('F:/DB2/DownSegZsc/DB2_s' + str(j) + 'allDownZsc.h5', 'r')
        #将六次重复手势分开存储
        Data0, Data1, Data2, Data3, Data4, Data5 = file['Data0'][:], file['Data1'][:], file['Data2'][:] ,file['Data3'][:], file['Data4'][:],  file['Data5'][:]
        Label0, Label1, Label2, Label3, Label4, Label5 = file['label0'][:], file['label1'][:], file['label2'][:] , file['label3'][:], file['label4'][:], file['label5'][:]
        file.close()

        #选定手势做训练集，测试集，验证集
        X_train = np.concatenate([Data0, Data2, Data3], axis=0)
        Y_train = np.concatenate([Label0, Label2, Label3], axis=0)

        X_vali = Data5
        Y_vali = Label5
        # X_vali = np.concatenate([Data1, Data4], axis=0)
        # Y_vali = np.concatenate([Label1, Label4], axis=0)

        Xtrain1, Xtrain2, Xtrain3,Xtrain4, Xtrain5, Xtrain6,Xtrain7, Xtrain8, Xtrain8,Xtrain10, Xtrain11, Xtrain12 = Sep12Data(X_train)
        Xvali1, Xvali2, Xvali3,Xvali4, Xvali5, Xvali6,Xvali7, Xvali8, Xvali9,Xvali10, Xvali11, Xvali12 = Sep12Data(X_vali)

        Y_train = tf.keras.utils.to_categorical(np.array(Y_train))
        Y_vali=tf.keras.utils.to_categorical(np.array(Y_vali))

        callbacks = [
            # EarlyStopping(monitor='val_accuracy', patience=5),
            ModelCheckpoint(filepath='F:/DB2/model/DownAway12reluBNCNN1D/111'
                                     '/DB2_s'+ str(j) + '6seg205mZsc.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model = DownAway12reluBNCNN1D()
        history = model.fit([Xtrain1, Xtrain2, Xtrain3,Xtrain4, Xtrain5, Xtrain6,Xtrain7, Xtrain8, Xtrain8,Xtrain10, Xtrain11, Xtrain12], Y_train, epochs=150, verbose=2, batch_size=32
            ,validation_data=([Xvali1, Xvali2, Xvali3,Xvali4, Xvali5, Xvali6,Xvali7, Xvali8, Xvali9,Xvali10, Xvali11, Xvali12],Y_vali), callbacks=callbacks)

        # loss= history.history['loss']
        # val_loss = history.history['val_loss']
        # accuracy =history.history['accuracy']
        # val_accuracy =history.history['val_accuracy']32
        # pltCurve(loss, val_loss, accuracy, val_accuracy)
        #早停机制保存最优模型后，不再另外保存
        # model.save('D:/Pengxiangdong/ZX/modelsave/3Channel/Away3CBAMcat/DB2_s'+str(j)+'re25seg400100mZsc.h5')
        # tf.keras.utils.plot_model(model, to_file='../ModelPng/Away3reluBNCBAMcatNEW.png', show_shapes=True)

