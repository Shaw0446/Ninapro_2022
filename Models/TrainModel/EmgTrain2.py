import tensorflow as tf
import h5py
import numpy as np
import nina_funcs as nf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from Models.DBEmgNet.CBAMFile import  Away3reluBNCBAMcat
from Util.SepData import Sep3Data
from tfdeterminism import patch
root_data='F:/DB2'

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

def restore(array):
    N1 = 4;
    N2 = 100
    X = np.zeros([array.shape[0], N1, N2, array.shape[2]])
    for i in range(len(array)):
        temp = array[i, :, :]
        for j in range(12):
            temp2 = temp[:,j].reshape(N1, N2)
            X[i, :, :, j] = temp2

    return X





if __name__ == '__main__':
    for j in range(1, 2):
        file = h5py.File(root_data + '/data/Comb_Seg/DB2_s' + str(j) + 'Seg17_zsc.h5', 'r')
        emg1, emg3, emg4, emg6 = file['x_train1'][:], file['x_train3'][:] \
            , file['x_train4'][:], file['x_train6'][:]
        y_train1, y_train3, y_train4, y_train6 = file['y_train1'][:], file['y_train3'][:], file['y_train4'][:] \
            , file['y_train6'][:]

        X_train = np.concatenate([emg1, emg3, emg4, emg6], axis=0)
        y_train = np.concatenate([y_train1, y_train3, y_train4, y_train6], axis=0)
        Y_train = nf.get_categorical(y_train)

        X_test = file['x_test'][:]
        # X_test = X_test.transpose((0,2,1))
        y_test = file['y_test'][:]
        Y_test = nf.get_categorical(y_test)
        file.close()

        Xtrain1, Xtrain2, Xtrain3 = Sep3Data(X_train)
        Xvali1, Xvali2, Xvali3 = Sep3Data(X_test)


        callbacks = [  # 1设置学习率衰减,2保存最优模型
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=root_data+'/DB2_model/DB2_s' + str(j) + 'model.h5'
                            , monitor='val_accuracy', save_best_only=True)]
        model = Away3reluBNCBAMcat()
        history = model.fit([Xtrain1, Xtrain2, Xtrain3], Y_train, epochs=50, verbose=2, batch_size=32
                            # , callbacks=callbacks)
                            , validation_data=([Xvali1, Xvali2, Xvali3], Y_test), callbacks=callbacks)

