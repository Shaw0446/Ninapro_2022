import numpy
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL

from Models.DBlayers.CBAM import cbam_time, cbam_acquisition

channel_axis = 1 if K.image_data_format() == "channels_first" else 3


# 3条支路，每个支路做串行的卷积注意力,在网络层加入标准化，注意顺序,早期网络加入1×1卷积
def EmgCNN():
    input1 = KL.Input(shape=(12, 400))
    input11 = tf.expand_dims(input=input1, axis=-1)


    #早期融合网络，加入1×1卷积
    x1 = KL.Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), activation='relu', padding='same')(input11)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Conv2D(filters=128, kernel_size=(2, 1), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)

    c = KL.GlobalAvgPool2D()(x1)
    X = KL.Dense(128, activation='relu')(c)
    X = KL.Dropout(0.1)(X)
    s = KL.Dense(17, activation='softmax')(X)
    model = tf.keras.Model(inputs=input1, outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def EmgCNN2():
    input1 = KL.Input(shape=(400, 12))
    input11 = tf.expand_dims(input=input1, axis=3)
    #早期融合网络，加入1×1卷积
    x1 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(2, 1), activation='relu', padding='same')(input11)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Conv2D(filters=128, kernel_size=(1, 2), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)

    c = KL.GlobalAvgPool2D()(x1)
    X = KL.Dense(128, activation='relu')(c)
    X = KL.Dropout(0.1)(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=input1, outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def EmgCNN3():
    input1 = KL.Input(shape=(400, 12))
    re = KL.Reshape(target_shape=(20, 20, 12))(input1)
    #早期融合网络，加入1×1卷积
    x1 = KL.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(re)
    x1 = KL.BatchNormalization()(x1)
    x1 = cbam_time(x1)
    x1 = KL.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)
    output1 = cbam_acquisition(x1)

    c = KL.GlobalAvgPool2D()(output1)
    X = KL.Dense(128, activation='relu')(c)
    X = KL.Dropout(0.1)(X)
    s = KL.Dense(17, activation='softmax')(X)
    model = tf.keras.Model(inputs=input1, outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def EmgAway3reluBConv():
    input1 = KL.Input(shape=(400, 8))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = KL.Input(shape=(400, 2))
    input21 = tf.expand_dims(input=input2, axis=3)
    input3 = KL.Input(shape=(400, 2))
    input31 = tf.expand_dims(input=input3, axis=3)

    x1 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(input11)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Conv2D(filters=128, kernel_size=(1, 8), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)
    output1 =x1

    x2 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(input21)
    x2 = KL.BatchNormalization()(x2)
    x2 = KL.Conv2D(filters=128, kernel_size=(1, 2), strides=(1, 1), activation='relu', padding='same')(x2)
    x2 = KL.BatchNormalization()(x2)
    output2 = x2

    x3 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(input31)
    x3 = KL.BatchNormalization()(x3)
    x3 = KL.Conv2D(filters=128, kernel_size=(1, 2), strides=(1, 1), activation='relu', padding='same')(x3)
    x3 = KL.BatchNormalization()(x3)
    output3 = x3

    c = KL.Concatenate(axis=-2)([output1, output2, output3])
    X = KL.GlobalAvgPool2D()(c)
    X = KL.Dense(512, activation='relu')(X)
    X = KL.Dropout(0.1)(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1, input2, input3], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


