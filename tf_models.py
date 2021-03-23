import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import numpy as np

def standard_conv():
    return layers.Conv2D

def depthwise_conv():
    return layers.SeparableConv2D

def get_autoencoder_model_s(conv, use_batchnorm = False, enlarge_by = 2):
    """
    This is smaller baseline model we designed that we use a prototype
    for GenSynth.
    """
    input_img = keras.Input(shape=(32,128,1))
    x = conv(4*enlarge_by, (3,3), padding = 'same')(input_img)
    x = layers.MaxPool2D(pool_size = (1,2))(x)
    x = layers.Activation('relu')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = conv(8*enlarge_by, (3,3), padding = 'same')(x)
    x = layers.MaxPool2D(pool_size = (1,2))(x)
    x = layers.Activation('relu')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = conv(16*enlarge_by, (3,3), padding = 'same')(x)
    encoded = layers.MaxPool2D(pool_size = (4,4))(x)

    x = layers.UpSampling2D(size = (4,4))(encoded)
    x = conv(16*enlarge_by, (3,3), padding = 'same')(x)
    x = layers.Activation('relu')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(size = (1,2))(x)
    x = conv(8*enlarge_by, (3,3), padding = 'same')(x)
    x = layers.Activation('relu')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(size = (1,2))(x)
    x = conv(4*enlarge_by, (3,3), padding = 'same')(x)
    x = layers.Activation('relu')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    decoded = conv(1, (3,3), padding = 'same')(x)
    # y = layers.Conv2D(16, (3,30))

    autoencoder = keras.Model(input_img, decoded)
    opt = keras.optimizers.Adam(lr = 0.001)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')
    return autoencoder

def get_autoencoder_m(latentDim=40):
    input_img = keras.Input(shape=(32,128,1))
    x = layers.Conv2D(4, (3,3), padding = 'same', strides=(1,2))(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(8, (3,3), padding = 'same', strides=(1,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(16, (3,3), padding = 'same', strides=(2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(32, (3,3), padding = 'same', strides=(2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, (3,3), padding = 'same', strides=(2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    volumeSize = keras.backend.int_shape(x)
    x = layers.Conv2D(latentDim, (4,4), strides=(1,1), padding='valid')(x)
    encoded = layers.Flatten()(x)

    x = layers.Dense(volumeSize[1] * volumeSize[2] * volumeSize[3])(encoded) 
    x = layers.Reshape((volumeSize[1], volumeSize[2], 64))(x)             

    x = layers.Conv2DTranspose(32, (3, 3),strides=(2,2), padding='same')(x)  #8x8
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(16, (3, 3),strides=(2,2), padding='same')(x)  #8x8
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(8, (3, 3),strides=(2,2), padding='same')(x)  #8x8
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(4, (3, 3),strides=(1,2), padding='same')(x)  #8x8
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    decoded = layers.Conv2DTranspose(1, (3, 3),strides=(1,2), padding='same')(x) 

    autoencoder = keras.Model(inputs=input_img, outputs=decoded)
    opt = keras.optimizers.Adam(lr = 0.01)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')
    return autoencoder

def conv_baseline(inputDim=(32,128), latentDim=40):
    """
    This is the convolutional autoencoder as described in https://arxiv.org/abs/2006.10417
    The implementation is from: https://github.com/APILASTRI/DCASE_Task2_UMINHO

    We move the keras implementation to tf.keras.

    We use this as a baseline for comparision 
    """
    input_img = keras.Input(shape=(inputDim[0], inputDim[1], 1))  # adapt this if using 'channels_first' image data format

    # encoder
    x = layers.Conv2D(32, (5, 5),strides=(1,2), padding='same')(input_img)   #32x128 -> 32x64
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (5, 5),strides=(1,2), padding='same')(x)           #32x32
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (5, 5),strides=(2,2), padding='same')(x)          #16x16
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3),strides=(2,2), padding='same')(x)          #8x8
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3),strides=(2,2), padding='same')(x)          #4x4
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    volumeSize = keras.backend.int_shape(x)
    # at this point the representation size is latentDim i.e. latentDim-dimensional
    x = layers.Conv2D(latentDim, (4,4), strides=(1,1), padding='valid')(x)
    encoded = layers.Flatten()(x)
    
    
    # decoder
    x = layers.Dense(volumeSize[1] * volumeSize[2] * volumeSize[3])(encoded) 
    x = layers.Reshape((volumeSize[1], volumeSize[2], 512))(x)                #4x4

    x = layers.Conv2DTranspose(256, (3, 3),strides=(2,2), padding='same')(x)  #8x8
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(128, (3, 3),strides=(2,2), padding='same')(x)  #16x16   
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(64, (5, 5),strides=(2,2), padding='same')(x)   #32x32
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(32, (5, 5),strides=(1,2), padding='same')(x)   #32x64
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    decoded = layers.Conv2DTranspose(1, (5, 5),strides=(1,2), padding='same')(x) 

    autoencoder = keras.Model(inputs=input_img, outputs=decoded)
    opt = keras.optimizers.Adam(lr = 0.001)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')
    return autoencoder
#########################################################################

def get_autoencoder_model(use_batchnorm = True):
    input_img = keras.Input(shape=(32, 128, 1))    # adapt this if using 'channels_first' image data format

    x = layers.Conv2D(32, (5, 5), strides = (1,2), padding='same')(input_img) # 32x128x1 -> 32x128x1
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (5, 5), strides = (1,2), padding='same')(x) # 16x64x32 -> 16x64x32
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (5, 5), strides = (2,2), padding='same')(x) # 8x32x32 -> 8x32x64
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), strides = (2,2), padding='same')(x) # 4x16x32 -> 4x16x128
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), strides = (2,2), padding='same')(x) # 4x16x32 -> 4x16x128
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    volume_size = keras.backend.int_shape(x)
    x = layers.Conv2D(40, (4,4), strides=(1,1), padding = 'valid')(x)  # 4x16x128 -> 4x16x40
    encoded = layers.Flatten()(x)

    x = layers.Dense(volume_size[1] * volume_size[2] * volume_size[3])(encoded) 
    x = layers.Reshape((volume_size[1], volume_size[2], 512))(x)     
    # at this point the representation is (6, 6, 128), i.e. 128-dimensional

    x = layers.Conv2DTranspose(256, (3, 3), strides = (2,2), padding='same')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides = (2,2), padding='same')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides = (2, 2), padding='same')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(32, (5, 5), strides = (1, 2), padding='same')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    decoded = layers.Conv2DTranspose(1, (5, 5), strides = (1,2), padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    opt = keras.optimizers.Adam(lr = 0.001)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')
    return autoencoder

def get_ds_autoencoder_model(use_batchnorm = True, h_space_flat = True):
    input_img = tf.keras.Input((32, 128, 1))    # adapt this if using 'channels_first' image data format

    x = layers.SeparableConv2D(32, (5, 5), padding='same')(input_img) # 32x128x1 -> 32x128x1
    x = layers.MaxPool2D(pool_size = (1,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(64, (5, 5),padding='same')(x) # 16x64x32 -> 16x64x32
    x = layers.MaxPool2D(pool_size = (1,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(128, (5, 5), padding='same')(x) # 8x32x32 -> 8x32x64
    x = layers.MaxPool2D(pool_size = (2,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(256, (3, 3), padding='same')(x) # 4x16x32 -> 4x16x128
    x = layers.MaxPool2D(pool_size = (2,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(512, (3, 3), padding='same')(x) # 4x16x32 -> 4x16x128
    x = layers.MaxPool2D(pool_size = (2,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    if h_space_flat:
        volume_size = keras.backend.int_shape(x)
        x = layers.SeparableConv2D(40, (4,4), strides=(1,1), padding = 'valid')(x)  # 4x16x128 -> 4x16x40
        encoded = layers.Flatten()(x)
        x = layers.Dense(volume_size[1] * volume_size[2] * volume_size[3])(encoded) 
        x = layers.Reshape((volume_size[1], volume_size[2], 512))(x)     
    else:
        x = layers.SeparableConv2D(40, (4,4), strides=(1,1), padding = 'valid')(x)  # 4x16x128 -> 4x16x40
        x = layers.UpSampling2D(size = (4,4))(x)
    # at this point the representation is (6, 6, 128), i.e. 128-dimensional

    x = layers.SeparableConv2D(256, (3, 3), padding='same')(x)
    x = layers.UpSampling2D(size = (2,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(128, (3, 3), padding='same')(x)
    x = layers.UpSampling2D(size = (2,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(64, (5, 5), padding='same')(x)
    x = layers.UpSampling2D(size = (2,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(32, (5, 5), padding='same')(x)
    x = layers.UpSampling2D(size = (1,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(1, (5, 5), padding='same')(x)
    decoded = layers.UpSampling2D(size = (1,2))(x)

    autoencoder = keras.Model(input_img, decoded)
    opt = keras.optimizers.Adam(lr = 0.001)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')
    return autoencoder

def get_ds_autoencoder_model2(use_batchnorm = True, h_space_flat = True):
    input_img = tf.keras.Input((32, 128, 1))    # adapt this if using 'channels_first' image data format

    x = layers.SeparableConv2D(32, (5, 5), padding='same')(input_img) # 32x128x1 -> 32x128x1
    x = layers.MaxPool2D(pool_size = (1,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(32, (5, 5),padding='same')(x) # 16x64x32 -> 16x64x32
    x = layers.MaxPool2D(pool_size = (1,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(32, (5, 5), padding='same')(x) # 8x32x32 -> 8x32x64
    x = layers.MaxPool2D(pool_size = (2,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(32, (3, 3), padding='same')(x) # 4x16x32 -> 4x16x128
    x = layers.MaxPool2D(pool_size = (2,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(32, (3, 3), padding='same')(x) # 4x16x32 -> 4x16x128
    x = layers.MaxPool2D(pool_size = (2,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    if h_space_flat:
        volume_size = keras.backend.int_shape(x)
        x = layers.SeparableConv2D(40, (4,4), strides=(1,1), padding = 'valid')(x)  # 4x16x128 -> 4x16x40
        encoded = layers.Flatten()(x)
        x = layers.Dense(volume_size[1] * volume_size[2] * volume_size[3])(encoded) 
        x = layers.Reshape((volume_size[1], volume_size[2], 512))(x)     
    else:
        x = layers.SeparableConv2D(20, (4,4), strides=(1,1), padding = 'valid')(x)  # 4x16x128 -> 4x16x40
        x = layers.UpSampling2D(size = (4,4))(x)
    # at this point the representation is (6, 6, 128), i.e. 128-dimensional

    x = layers.SeparableConv2D(32, (3, 3), padding='same')(x)
    x = layers.UpSampling2D(size = (2,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(32, (3, 3), padding='same')(x)
    x = layers.UpSampling2D(size = (2,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(32, (5, 5), padding='same')(x)
    x = layers.UpSampling2D(size = (2,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(32, (5, 5), padding='same')(x)
    x = layers.UpSampling2D(size = (1,2))(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(1, (5, 5), padding='same')(x)
    decoded = layers.UpSampling2D(size = (1,2))(x)

    autoencoder = keras.Model(input_img, decoded)
    opt = keras.optimizers.Adam(lr = 0.01)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')
    return autoencoder

def baseline_keras_model(inputDim):
    """
    From: https://github.com/MIMII-hitachi/mimii_baseline/
    define the keras model
    the model based on the simple dense auto encoder (64*64*8*64*64)
    """
    inputLayer = layers.Input(shape=(inputDim,))
    h = layers.Dense(64, activation="relu")(inputLayer)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dense(8, activation="relu")(h)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dense(inputDim, activation=None)(h)

    autoencoder = keras.Model(inputLayer, h)
    opt = keras.optimizers.Adam(lr = 0.01)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')
    return autoencoder


if __name__ == "__main__":

    model = get_autoencoder_m()
    model.summary()

    # keras.models.save_model(model, "tinyanomaly_baseline", save_format='tf')