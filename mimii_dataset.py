import tensorflow as tf
import numpy as np
import os
import sklearn.preprocessing

_DATA_DIR = "features"
_TRAIN_BATCH_SIZE = 32
_STEP = 32

class MIMIIDataset:
    def __init__(self):
        self.data_dir = _DATA_DIR
        self.train_batch_size = 1
        self.normalizer = sklearn.preprocessing.StandardScaler()
        self.step = _STEP

    def get_train_dataset(self, num_shards=1, shard_index=0):
        self.dim = (32,128)
        x_train = np.load(os.path.join(self.data_dir,"train_6dB_fan_id_06.npy"))
        # nsamples, nx, ny = x_train.shape
        # x_train_flat = x_train.reshape((nsamples,nx*ny))
        # x_train_flat = self.normalizer.fit_transform(x_train_flat)
        # x_train = x_train_flat.reshape((nsamples, nx, ny))
        x_train = x_train / np.max(x_train)

        self.indexes_start = np.arange(x_train.shape[1]-self.dim[0]+self.step, step=self.step)
        self.max = len(self.indexes_start)
        self.indexes = np.arange(x_train.shape[0])
        self.indexes = np.repeat(self.indexes, self.max)
        self.indexes_start = np.repeat(self.indexes_start, x_train.shape[0])
        np.random.shuffle(self.indexes)
        np.random.shuffle(self.indexes_start)

        num_batches = int(np.floor(x_train.shape[0] * self.max  / self.train_batch_size))

        x_train_cropped = np.empty((num_batches,*self.dim))
        for index in range(num_batches):
            indexes = self.indexes[index*self.train_batch_size:(index+1)*self.train_batch_size]
            indexes_start = self.indexes_start[index*self.train_batch_size:(index+1)*self.train_batch_size]
            x = self.__data_generation(x_train, indexes, indexes_start)
            x_train_cropped[index*self.train_batch_size:(index+1)*self.train_batch_size,:] = self.__data_generation(x_train, indexes, indexes_start)
        
        self.train_batch_size = 32
        x_train_cropped  = np.expand_dims(x_train_cropped,-1)
        print(x_train_cropped.shape)
        trainset = tf.data.Dataset.from_tensor_slices((x_train_cropped, x_train_cropped)).batch(self.train_batch_size)
        trainset = trainset.shuffle(self.train_batch_size)
        return trainset
    
    def get_test_dataset(self, num_shards=1, shard_index=0):
        x_test = np.load(os.path.join(self.data_dir,"test_6dB_fan_id_00.npy"))
        nsamples, nx, ny = x_test.shape
        x_test_flat = x_test.reshape((nsamples,nx*ny))
        x_test_flat = self.normalizer.transform(x_test_flat)
        x_test = x_test_flat.reshape((nsamples, nx, ny))

        testset = tf.data.Dataset.from_tensor_slices((x_test,x_test)).batch(self.train_batch_size)
        return testset

    def __data_generation(self, data, indexes, index_start):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.train_batch_size, *self.dim))

        # Generate data
        for i, (id_file, id_start) in enumerate(zip(indexes, index_start)):

            x = data[id_file,]
            length, mels = x.shape

            start = id_start

            start = min(start, length - self.dim[0])
            
            # crop part of sample
            crop = x[start:start+self.dim[0], :]

            X[i,] = crop
        return X