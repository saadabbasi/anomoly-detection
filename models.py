from keras.models import Model
from keras.layers import Input, Reshape, Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import UpSampling2D, BatchNormalization, Activation, Conv2DTranspose
import keras.layers as layers
from keras.backend import int_shape
import keras

def keras_model(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder (64*64*8*64*64)
    """
    inputLayer = Input(shape=(inputDim,))
    h = Dense(128, activation="relu")(inputLayer)
    h = Dense(64, activation="relu")(h)
    h = Dense(8, activation="relu")(h)
    h = Dense(64, activation="relu")(h)
    h = Dense(128, activation="relu")(h)
    h = Dense(inputDim, activation=None)(h)

    return Model(inputs=inputLayer, outputs=h)


def get_autoencoder_model():
    input_img = Input(shape=(32, 128, 1))    # adapt this if using 'channels_first' image data format

    x = Conv2D(32, (5, 5), strides = (1,2), padding='same')(input_img) # 32x128x1 -> 32x128x1
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (5, 5), strides = (1,2), padding='same')(x) # 16x64x32 -> 16x64x32
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (5, 5), strides = (2,2), padding='same')(x) # 8x32x32 -> 8x32x64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), strides = (2,2), padding='same')(x) # 4x16x32 -> 4x16x128
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), strides = (2,2), padding='same')(x) # 4x16x32 -> 4x16x128
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    volume_size = int_shape(x)
    x = Conv2D(40, (4,4), strides=(1,1), padding = 'valid')(x)  # 4x16x128 -> 4x16x40
    encoded = Flatten()(x)

    x = Dense(volume_size[1] * volume_size[2] * volume_size[3])(encoded) 
    x = Reshape((volume_size[1], volume_size[2], 512))(x)     
    # at this point the representation is (6, 6, 128), i.e. 128-dimensional

    x = Conv2DTranspose(256, (3, 3), strides = (2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(128, (3, 3), strides = (2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (5, 5), strides = (2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(32, (5, 5), strides = (1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    decoded = Conv2DTranspose(1, (5, 5), strides = (1,2), padding='same')(x)

    autoencoder = Model(input_img, decoded)
    opt = keras.optimizers.Adam(lr = 0.01)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')
    return autoencoder

def get_model(inputDim, latentDim):
    """
    define the keras model
    the model based on the simple convolutional auto encoder 
    """
    input_img = Input(shape=(inputDim[0], inputDim[1], 1))  # adapt this if using 'channels_first' image data format

    # encoder
    x = Conv2D(32, (5, 5),strides=(1,2), padding='same')(input_img)   #32x128 -> 32x64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (5, 5),strides=(1,2), padding='same')(x)           #32x32
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (5, 5),strides=(2,2), padding='same')(x)          #16x16
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3),strides=(2,2), padding='same')(x)          #8x8
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3),strides=(2,2), padding='same')(x)          #4x4
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    volumeSize = int_shape(x)
    # at this point the representation size is latentDim i.e. latentDim-dimensional
    x = Conv2D(latentDim, (4,4), strides=(1,1), padding='valid')(x)
    encoded = Flatten()(x)
    
    
    # decoder
    x = Dense(volumeSize[1] * volumeSize[2] * volumeSize[3])(encoded) 
    x = Reshape((volumeSize[1], volumeSize[2], 512))(x)                #4x4

    x = Conv2DTranspose(256, (3, 3),strides=(2,2), padding='same')(x)  #8x8
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(128, (3, 3),strides=(2,2), padding='same')(x)  #16x16   
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (5, 5),strides=(2,2), padding='same')(x)   #32x32
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(32, (5, 5),strides=(1,2), padding='same')(x)   #32x64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    decoded = Conv2DTranspose(1, (5, 5),strides=(1,2), padding='same')(x) 
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder