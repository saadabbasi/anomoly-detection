import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import numpy as np

def standard_conv():
    return layers.Conv2D

def depthwise_conv():
    return layers.SeparableConv2D

def get_autoencoder_model_s(conv, use_batchnorm = False):
    input_img = keras.Input(shape=(32,128,1))

    x = conv(4*2, (3,3), padding = 'same')(input_img)
    x = layers.MaxPool2D(pool_size = (1,2))(x)
    x = layers.Activation('relu')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = conv(8*2, (3,3), padding = 'same')(x)
    x = layers.MaxPool2D(pool_size = (1,2))(x)
    x = layers.Activation('relu')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = conv(16*2, (3,3), padding = 'same')(x)
    encoded = layers.MaxPool2D(pool_size = (4,4))(x)

    x = layers.UpSampling2D(size = (4,4))(encoded)
    x = conv(16*2, (3,3), padding = 'same')(x)
    x = layers.Activation('relu')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(size = (1,2))(x)
    x = conv(8*2, (3,3), padding = 'same')(x)
    x = layers.Activation('relu')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(size = (1,2))(x)
    x = conv(4*2, (3,3), padding = 'same')(x)
    x = layers.Activation('relu')(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    decoded = conv(1, (3,3), padding = 'same')(x)
    # y = layers.Conv2D(16, (3,30))

    autoencoder = keras.Model(input_img, decoded)
    opt = keras.optimizers.Adam(lr = 0.001)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')
    return autoencoder

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
    import tensorflow as tf
    import tensorflow.keras.backend as K
    # from keras.applications.mobilenet import MobileNet

    run_meta = tf.RunMetadata()
    with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)
        # net = MobileNet(alpha=.75, input_tensor=tf.placeholder('float32', shape=(1,32,32,3)))
        # net = baseline_keras_model(64*5)
        net = get_autoencoder_model_s(standard_conv(),use_batchnorm=True)



    model = baseline_keras_model(64*5)
    model.summary()

    keras.models.save_model(model, "tinyanomaly_baseline", save_format='tf')

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset, batch_size=32, dim=(32,128), shuffle=True, step=8):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = dataset
        
        self.step = step
        self.indexes_start = np.arange(self.data.shape[1]-self.dim[0]+self.step, step=self.step)
        self.max = len(self.indexes_start)
        self.indexes = np.arange(self.data.shape[0])
        
        self.indexes = np.repeat(self.indexes, self.max )
        self.indexes_start = np.repeat(self.indexes_start, self.data.shape[0])
    
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data.shape[0] * self.max  / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        indexes_start = self.indexes_start[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X = self.__data_generation(indexes, indexes_start).reshape((self.batch_size, *self.dim, 1))

        return X, X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.indexes_start)


    def __data_generation(self, indexes, index_start):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, (id_file, id_start) in enumerate(zip(indexes, index_start)):

            x = self.data[id_file,]
            length, mels = x.shape

            start = id_start

            start = min(start, length - self.dim[0])
            
            # crop part of sample
            crop = x[start:start+self.dim[0], :]

            X[i,] = crop
        return X