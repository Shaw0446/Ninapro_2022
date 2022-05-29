import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL

from Models.PopularModel.Away3CBAM import cbam_acquisition, cbam_time
from Models.PopularModel.FeaModel import cbam_module


def FeaAway3CBAM():
    inputh1 = KL.Input(shape=(16, 16, 8))
    inputh2 = KL.Input(shape=(16, 16, 2))
    inputh3 = KL.Input(shape=(16, 16, 2))
    inputl1 = KL.Input(shape=(16, 16, 12))

    x1 = KL.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(inputh1)
    x1 = KL.BatchNormalization()(x1)
    x1 = cbam_module(x1)
    x1 = KL.Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)
    output1 = cbam_module(x1)

    x2 = KL.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(inputh2)
    x2 = KL.BatchNormalization()(x2)
    x2 =cbam_module(x2)
    x2 = KL.Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), activation='relu', padding='same')(x2)
    x2 = KL.BatchNormalization()(x2)
    output2 =cbam_module(x2)

    x3 = KL.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(inputh3)
    x3 = KL.BatchNormalization()(x3)
    x3 = cbam_module(x3)
    x3 = KL.Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), activation='relu', padding='same')(x3)
    x3 = KL.BatchNormalization()(x3)
    output3 = cbam_module(x3)

    x4 = KL.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(inputl1)
    x4 = KL.BatchNormalization()(x4)
    x4 = cbam_module(x4)
    x4 = KL.Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), activation='relu', padding='same')(x4)
    x4 = KL.BatchNormalization()(x4)
    output4 = cbam_module(x1)

    c = KL.Concatenate(axis=-2)([output1, output2, output3])
    X = KL.GlobalAvgPool2D()(c)
    X = KL.Dense(128, activation='relu')(X)
    X = KL.Dropout(0.1)(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[inputh1, inputh2, inputh3, inputl1], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def FeaAndEmg_model1():
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

    x2 = KL.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(input2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='valid')(x2)
    x2 = KL.Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', padding='valid')(x2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='valid')(x2)
    x2 = KL.GlobalAvgPool1D()(x2)


    X = KL.Concatenate(axis=-1)([x1,x2])
    X = KL.Dense(256, activation='relu')(X)
    X = KL.Dropout(0.2)(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1,input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

