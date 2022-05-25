import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL

channel_axis = 1 if K.image_data_format() == "channels_first" else 3


# CAM 特征通道注意力
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal',
                         use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True,
                         bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])


# SAM 空间注意力
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(2, 2), padding="same", activation='sigmoid',
                     kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)

# 标准的CBAM，没有卷积！！！加入ResBlock
def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])


def Stage1():
    input00 = KL.Input(shape=(20, 12))
    input0 = tf.expand_dims(input=input00, axis=3)
    input1 = KL.Input(shape=(16, 16, 12))
    input2 = KL.Input(shape=(16, 16, 12))

    x0 = KL.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(input0)
    x0 = KL.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x0)
    x0 = KL.LocallyConnected2D(64, (1, 1))(x0)
    x0 = KL.LocallyConnected2D(64, (1, 1))(x0)
    x0 = KL.Flatten()(x0)

    x1 = KL.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(input1)
    x1 = KL.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = KL.LocallyConnected2D(64, (1,1))(x1)
    x1 = KL.LocallyConnected2D(64, (1,1))(x1)
    x1 = KL.Flatten()(x1)

    x2 = KL.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(input1)
    x2 = KL.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x2)
    x2 = KL.LocallyConnected2D(64, (1, 1))(x2)
    x2 = KL.LocallyConnected2D(64, (1, 1))(x2)
    x2 = KL.Flatten()(x2)

    c = KL.Add()([x1, x2])
    c = KL.Concatenate()([c, x0])
    X = KL.Dense(512, activation='relu')(c)
    X = KL.Dense(256, activation='relu')(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input00, input1, input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def Stage2():
    input1 = KL.Input(shape=(16, 16, 12))
    input2 = KL.Input(shape=(16, 16, 12))

    x1 = KL.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(input1)
    x1 = KL.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = KL.LocallyConnected2D(64, (1,1))(x1)
    x1 = KL.LocallyConnected2D(64, (1,1))(x1)
    x1 = KL.Dropout(0.1)(x1)
    x1 = KL.Dense(256, activation='relu')(x1)

    x2 = KL.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(input1)
    x2 = KL.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x2)
    x2 = KL.LocallyConnected2D(64, (1, 1))(x2)
    x2 = KL.LocallyConnected2D(64, (1, 1))(x2)
    x2 = KL.Dropout(0.1)(x2)
    x2 = KL.Dense(256, activation='relu')(x2)

    c = KL.Add()([x1, x2])
    X = KL.Dense(256, activation='relu')(c)
    X = KL.GlobalAvgPool2D()(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1, input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


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



def model1():
    input1 = KL.Input(shape=(36,1))

    x1 = KL.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(input1)
    x1 = KL.MaxPool1D(pool_size=2, strides=1, padding='valid')(x1)
    x1 = KL.Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', padding='valid')(x1)
    x1 = KL.MaxPool1D(pool_size=2, strides=1, padding='valid')(x1)
    x1 = KL.Dropout(0.1)(x1)
    x1 = KL.Dense(256, activation='relu')(x1)
    X = KL.Dropout(0.2)(x1)
    X = KL.GlobalAvgPool1D()(X)
    s = KL.Dense(17, activation='softmax')(X)
    model = tf.keras.Model(inputs=input1, outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model