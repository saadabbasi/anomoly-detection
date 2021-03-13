import tensorflow as tf
import numpy as np
import os
import sklearn.preprocessing
import sklearn.utils

# _NORMAL_DATA_DIR = "/gensynth/workspace/datasets/mimii_spectograms/normal"
# _ABNORMAL_DATA_DIR = "/gensynth/workspace/datasets/mimii_spectograms/abnormal"
_NORMAL_DATA_DIR = "spectograms/normal"
_ABNORMAL_DATA_DIR = "spectograms/abnormal"
_TRAIN_BATCH_SIZE = 128
_TEST_BATCH_SIZE = 512

def preprocess(input_img, target_img, y_true):
    input_img = input_img / tf.reduce_max(input_img)
    target_img = target_img / tf.reduce_max(target_img)
    return {'input_image': input_img, 'target_img': target_img, 'y_true': y_true}

def split_array(arr):
    middle_idx = len(arr) // 2
    a = arr[:middle_idx]
    b = arr[middle_idx:]
    return a,b

class MIMIIDataset():
    def __init__(self):
        self.normal_data_dir = _NORMAL_DATA_DIR
        self.abnormal_data_dir = _ABNORMAL_DATA_DIR
        self.train_batch_size = _TRAIN_BATCH_SIZE
        self.test_batch_size = _TEST_BATCH_SIZE
        # self.normalizer = sklearn.preprocessing.StandardScaler()
        self.dim = (32,128)

        normal = np.load(os.path.join(self.normal_data_dir,"normal_6dB_fan_id_06.npy")).astype(np.float32)
        abnormal = np.load(os.path.join(self.abnormal_data_dir,"abnormal_6dB_fan_id_06.npy")).astype(np.float32)

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

        # samples, nx, ny, _ = self.train_x.shape
        # train_x_flat = self.train_x.reshape((-1,nx*ny))
        # self.train_x = self.normalizer.fit_transform(train_x_flat).reshape((samples, nx, ny, 1))

        # samples, nx, ny, _ = self.test_x.shape
        # test_x_flat = self.test_x.reshape((-1,nx*ny))
        # self.test_x = self.normalizer.transform(test_x_flat).reshape((samples, nx, ny, 1))

        self.test_x, self.test_y = sklearn.utils.shuffle(self.test_x, self.test_y)
        self.test_x, self.val_x = split_array(self.test_x)
        self.test_y, self.val_y = split_array(self.test_y)

    def get_train_dataset(self, num_shards=1, shard_index=0):
        train_x = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_x, self.train_y)).batch(self.train_batch_size)
        train_x = train_x.map(preprocess)
        train_x = train_x.shuffle(32)
        return train_x, len(self.train_x), self.train_batch_size

    def get_validation_dataset(self, num_shards=1, shard_index=0):
        # have to move actual validation set
        val_x = tf.data.Dataset.from_tensor_slices((self.val_x, self.val_x, self.val_y)).batch(self.test_batch_size)
        val_x = val_x.map(preprocess)
        return val_x, len(self.val_x), self.test_batch_size

    def get_test_dataset(self):
        test_x = tf.data.Dataset.from_tensor_slices((self.test_x, self.test_x, self.test_y)).batch(self.test_batch_size)
        test_x = test_x.map(preprocess)
        return test_x, len(self.test_x), self.test_batch_size

