from os import makedirs
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import numpy as np
import os
# from keras_flops import get_flops

def standard_conv():
    return layers.Conv2D

def depthwise_conv():
    return layers.SeparableConv2D

def get_autoencoder_model_s(conv, use_batchnorm = False, enlarge_by = 2):
    """
    This is smaller baseline model we designed that we use a prototype
    for GenSynth.

    Used for fan machine type
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

def get_autoencoder_m(latentDim=60):
    """
    This is the medium sized autoencoder we built. Used for slider machine type.
    """
    input_img = keras.Input(shape=(32,128,1))
    x = layers.Conv2D(4, (3,3), padding = 'same', strides=(1,2))(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x) # 32 x 64

    x = layers.Conv2D(8, (3,3), padding = 'same', strides=(1,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x) # 32 x 32

    x = layers.Conv2D(16, (3,3), padding = 'same', strides=(2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x) # 16 x 16

    x = layers.Conv2D(32, (3,3), padding = 'same', strides=(2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x) # 8 x 8 

    x = layers.Conv2D(64, (3,3), padding = 'same', strides=(2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x) # 4 x 4

    volumeSize = keras.backend.int_shape(x)
    x = layers.Conv2D(latentDim, (4,4), strides=(1,1), padding='valid')(x)
    encoded = layers.Flatten()(x)

    x = layers.Dense(volumeSize[1] * volumeSize[2] * volumeSize[3])(encoded) 
    x = layers.Reshape((volumeSize[1], volumeSize[2], 64))(x)             

    x = layers.UpSampling2D(size = (2,2))(x)
    x = layers.Conv2D(32, (3,3), strides = (1,1), padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D(size = (2,2))(x)
    x = layers.Conv2D(16, (3,3), strides = (1,1), padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D(size = (2,2))(x)
    x = layers.Conv2D(8, (3,3), strides = (1,1), padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D(size = (1,2))(x)
    x = layers.Conv2D(4, (3,3), strides = (1,1), padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D(size = (1,2))(x)
    decoded = layers.Conv2D(1, (3,3), strides=(1,1), padding = 'same')(x)

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

def conv_baseline_90k(inputDim=(32,128), latentDim=40):
    input_img = keras.Input(shape=(inputDim[0], inputDim[1], 1))  # adapt this if using 'channels_first' image data format

    # encoder
    x = layers.Conv2D(16, (5, 5),strides=(1,2), padding='same')(input_img)   #32x128 -> 32x64
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (5, 5),strides=(1,2), padding='same')(x)           #32x32
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (5, 5),strides=(2,2), padding='same')(x)          #16x16
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3),strides=(2,2), padding='same')(x)          #8x8
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3),strides=(2,2), padding='same')(x)          #4x4
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    volumeSize = keras.backend.int_shape(x)
    # at this point the representation size is latentDim i.e. latentDim-dimensional
    x = layers.Conv2D(latentDim, (4,4), strides=(1,1), padding='valid')(x)
    encoded = layers.Flatten()(x)
    
    
    # decoder
    x = layers.Dense(volumeSize[1] * volumeSize[2] * volumeSize[3])(encoded) 
    x = layers.Reshape((volumeSize[1], volumeSize[2], 128))(x)                #4x4

    x = layers.Conv2DTranspose(128, (3, 3),strides=(2,2), padding='same')(x)  #8x8
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(64, (3, 3),strides=(2,2), padding='same')(x)  #16x16   
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(32, (5, 5),strides=(2,2), padding='same')(x)   #32x32
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(16, (5, 5),strides=(1,2), padding='same')(x)   #32x64
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    decoded = layers.Conv2DTranspose(1, (5, 5),strides=(1,2), padding='same')(x) 

    autoencoder = keras.Model(inputs=input_img, outputs=decoded)
    opt = keras.optimizers.Adam(lr = 0.001)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')
    return autoencoder

def conv_baseline_80k(inputDim=(32,128), latentDim=60):
    input_img = keras.Input(shape=(inputDim[0], inputDim[1], 1))  # adapt this if using 'channels_first' image data format
    x = layers.Conv2D(2, (5, 5),strides=(1,2), padding='same')(input_img)   #32x128 -> 32x64
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(4, (5, 5),strides=(2,2), padding='same')(x)           #32x32
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(8, (5, 5),strides=(2,2), padding='same')(x)          #16x16
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)

    volumeSize = keras.backend.int_shape(x)
    # at this point the representation size is latentDim i.e. latentDim-dimensional
    x = layers.Conv2D(latentDim, (4,4), strides=(1,1), padding='valid')(x)
    encoded = layers.Flatten()(x)
    
    
    # decoder
    x = layers.Dense(volumeSize[1] * volumeSize[2] * volumeSize[3])(encoded) 
    x = layers.Reshape((volumeSize[1], volumeSize[2], 8))(x)                #4x4


    x = layers.Conv2DTranspose(8, (3, 3),strides=(2,2), padding='same')(x)  #16x16   
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(4, (5, 5),strides=(2,2), padding='same')(x)   #32x32
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(2, (5, 5),strides=(2,2), padding='same')(x)   #32x64
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    decoded = layers.Conv2DTranspose(1, (5, 5),strides=(1,2), padding='same')(x) 

    autoencoder = keras.Model(inputs=input_img, outputs=decoded)
    opt = keras.optimizers.Adam(lr = 0.001)
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

def save_model_as_metagraph(model):
    sess = keras.backend.get_session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, 'tiny_anomoly_sc_m/model')

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
    tf.train.write_graph(frozen_graph, "baseline_84k", "baseline.pb", as_text=False)



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # model = get_autoencoder_model_s()
    # model = conv_baseline()
    model = conv_baseline_90k()
    model.summary()

    # flops = get_flops(model, batch_size=1)
    # print(flops/1e6)

    # # save_model_as_metagraph(model)

    # keras.models.save_model(model, 'tiny_anomoly_sc_m.h5', save_format='h5')
    freeze_session(keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
