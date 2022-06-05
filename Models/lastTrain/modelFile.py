import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL

from Models.PopularModel.Away3CBAM import cbam_acquisition, cbam_time
from Models.PopularModel.FeaModel import cbam_module
from Models.lastTrain.daNet import danet_resnet101



def FeaAndEmg_model2():
    input1 = KL.Input(shape=(400,12))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = KL.Input(shape=(36, 1))

    # 早期融合网络，加入1×1卷积
    x1 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(input11)
    x1 = KL.BatchNormalization()(x1)
    x1 = cbam_time(x1)
    x1 = KL.Conv2D(filters=128, kernel_size=(1, 8), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)
    x1 = cbam_acquisition(x1)
    x1 = KL.GlobalAvgPool2D()(x1)
    # x1 = danet_resnet101(input11, 17)



    x2 = KL.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(input2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='valid')(x2)
    x2 = KL.Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', padding='valid')(x2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='valid')(x2)
    x2 = KL.GlobalAvgPool1D()(x2)


    X = KL.Concatenate(axis=-1)([x1,x2])
    X = KL.Dense(256, activation='relu')(X)
    X = KL.Dropout(0.2)(X)
    s = KL.Dense(17, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1,input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model