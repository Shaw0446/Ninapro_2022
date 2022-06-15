import tensorflow as tf
from tensorflow.keras import layers as KL

from Models.DBlayers.CBAM import cbam_acquisition, cbam_time
from Models.DBlayers.SENet import se_block
from Models.DBlayers.daNet import danet_resnet101


def FeaAndEmg_model1():
    input1 = KL.Input(shape=(400,12))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = KL.Input(shape=(36, 1))

    # 早期融合网络，加入1×1卷积
    x1 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(input11)
    x1 = KL.BatchNormalization()(x1)
    # x1 = cbam_time(x1)
    x1 = KL.Conv2D(filters=128, kernel_size=(1, 8), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)
    # x1 = cbam_acquisition(x1)
    x1 = KL.GlobalAvgPool2D()(x1)



    x2 = KL.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(input2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='same')(x2)
    x2 = KL.Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', padding='same')(x2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='same')(x2)
    x2 = KL.GlobalAvgPool1D()(x2)


    X = KL.Concatenate(axis=-1)([x1,x2])
    X = KL.Dense(256, activation='relu')(X)
    X = KL.Dropout(0.2)(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1,input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def FeaAndEmgsoft_model2():
    input1 = KL.Input(shape=(400,12))
    # input11 = tf.expand_dims(input=input1, axis=3)
    input11 = KL.Reshape(target_shape=(20, 20, 12))(input1)
    input2 = KL.Input(shape=(36, 1))

    # 早期融合网络，加入1×1卷积
    n, row, col, channels =  input11.shape
    s1 = danet_resnet101(row, col, channels, 17)


    x2 = KL.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(input2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='valid')(x2)
    x2 = KL.Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', padding='valid')(x2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='valid')(x2)
    x2 = KL.GlobalAvgPool1D()(x2)
    X = KL.Dense(256, activation='relu')(x2)
    X = KL.Dropout(0.2)(X)
    s2 = KL.Dense(17, activation='softmax')(X)

    s = KL.Add()([s1, s2])
    model = tf.keras.Model(inputs=[input1,input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def FeaAndEmg_se():
    input1 = KL.Input(shape=(400,12))
    input11 = tf.expand_dims(input=input1, axis=3)
    input11 = KL.Reshape(target_shape=(20, 20, 12))(input11)
    input2 = KL.Input(shape=(36, 1))

    # 早期融合网络，加入1×1卷积

    x1 = KL.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input11)
    x1 = KL.Activation('relu')(x1)
    x1 = KL.BatchNormalization()(x1)
    x1 = se_block(x1)
    x1 = KL.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1),  padding='same')(x1)
    x1 = KL.Activation('relu')(x1)
    x1 = KL.BatchNormalization()(x1)
    x1 = se_block(x1)
    x1 = KL.GlobalAvgPool2D()(x1)


    x2 = KL.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(input2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='same')(x2)
    x2 = KL.Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', padding='same')(x2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='same')(x2)
    x2 = KL.GlobalAvgPool1D()(x2)
    s = KL.Add()([x1, x2])

    X = KL.Dense(256, activation='relu')(s)
    X = KL.Dropout(0.2)(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1, input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model