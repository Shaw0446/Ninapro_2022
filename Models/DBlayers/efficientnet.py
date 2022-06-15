import tensorflow as tf
from tensorflow.keras import layers as KL
import numpy as np

from tfdeterminism import patch
#确定随机数
patch()
np.random.seed(123)
tf.random.set_seed(123)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dir='F:/DB2'
# tf.keras.applications.efficientnet




def efficientnet_model1():
    input1 = KL.Input(shape=(400,12))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = KL.Input(shape=(36, 1))

    # 早期融合网络，加入1×1卷积
    x1 = KL.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(input11)
    x1 = tf.keras.applications.efficientnet(x1)
    x1 = KL.GlobalAvgPool2D()(x1)

    x2 = KL.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(input2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='valid')(x2)
    x2 = KL.Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', padding='same')(x2)
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
