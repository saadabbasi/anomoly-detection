import tensorflow as tf
import numpy as np
import os
import sklearn.preprocessing

_NORMAL_DATA_DIR = "spectograms/normal"
_ABNORMAL_DATA_DIR = "spectograms/abnormal"
_TRAIN_BATCH_SIZE = 32

class MIMIIDataset:
    def __init__(self):
        self.normal_data_dir = _NORMAL_DATA_DIR
        self.abnormal_data_dir = _ABNORMAL_DATA_DIR
        self.train_batch_size = 128
        self.test_batch_size = 512
        self.normalizer = sklearn.preprocessing.StandardScaler()
        self.dim = (32,128)

        self._read_dataset()
        samples, nx, ny, _ = self.train_x.shape
        train_x_flat = self.train_x.reshape((-1,nx*ny))
        self.train_x = self.normalizer.fit_transform(train_x_flat).reshape((samples, nx, ny, 1))

        samples, nx, ny, _ = self.test_x.shape
        test_x_flat = self.test_x.reshape((-1,nx*ny))
        self.test_x = self.normalizer.transform(test_x_flat).reshape((samples, nx, ny, 1))

    def get_train_dataset(self, num_shards=1, shard_index=0):
        train_x = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_x)).batch(self.train_batch_size)
        train_x = train_x.shuffle(32)
        return train_x, self.train_y

    def get_test_dataset(self, num_shards=1, shard_index=0):
        test_x = tf.data.Dataset.from_tensor_slices((self.test_x, self.test_x)).batch(self.test_batch_size)
        return test_x, self.test_y

    def _read_dataset(self):
        normal = np.load(os.path.join(self.normal_data_dir,"normal_6dB_fan_id_06.npy"))
        abnormal = np.load(os.path.join(self.abnormal_data_dir,"abnormal_6dB_fan_id_06.npy"))

        normal_labels = np.zeros((len(normal)))
        abnormal_labels = np.ones((len(abnormal)))
        train_x = normal[abnormal.shape[0]:]
        train_y = normal_labels[abnormal.shape[0]:]
        test_x = normal[:abnormal.shape[0]]
        test_y = normal_labels[:abnormal.shape[0]]
        test_x = np.concatenate((test_x, abnormal))
        test_y = np.concatenate((test_y, abnormal_labels))
        
        self.train_x = np.expand_dims(train_x,axis=-1)
        self.test_x = np.expand_dims(test_x,axis=-1)
        self.train_y = train_y
        self.test_y = test_y

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