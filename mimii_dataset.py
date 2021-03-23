import tensorflow as tf
import numpy as np
import os
import sklearn.preprocessing
import sklearn.utils

_SPECTOGRAM_DATA_DIR = "/gensynth/workspace/datasets/mimii_spectograms"

def preprocess(input_img, target_img, y_true):
    return {'input_image': input_img, 'target_img': target_img, 'y_true': y_true}

def split_array(arr):
    middle_idx = len(arr) // 2
    a = arr[:middle_idx]
    b = arr[middle_idx:]
    return a,b

class MIMIIDataset():
    def __init__(self, machine_type_id = "6dB_fan_id_06", spectogram_dir = 'spectograms', train_batch_size = 128, test_batch_size = 512):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        normal = np.load(os.path.join(spectogram_dir,"normal",f"normal_{machine_type_id}.npy")).astype(np.float32)
        abnormal = np.load(os.path.join(spectogram_dir,"abnormal",f"abnormal_{machine_type_id}.npy")).astype(np.float32)

        self.train_x, self.train_y, self.test_x, self.test_y = self._make_labels_and_combine(normal, abnormal)
        self._scale_dataset()

        self.test_x, self.test_y = sklearn.utils.shuffle(self.test_x, self.test_y)
        self.test_x, self.val_x = split_array(self.test_x)
        self.test_y, self.val_y = split_array(self.test_y)

    def train_dataset(self):
        return self.train_x, self.train_y

    def test_dataset(self):
        return self.test_x, self.test_y

    def val_dataset(self):
        return self.val_x, self.val_y

    def _scale_dataset(self):
        norm = sklearn.preprocessing.StandardScaler()
        samples, nx, ny, _ = self.train_x.shape
        train_x_flat = self.train_x.reshape((-1,nx*ny))
        self.train_x = norm.fit_transform(train_x_flat).reshape((samples, nx, ny, 1))

        samples, nx, ny, _ = self.test_x.shape
        test_x_flat = self.test_x.reshape((-1,nx*ny))
        self.test_x = norm.transform(test_x_flat).reshape((samples, nx, ny, 1))

    def _make_labels_and_combine(self, normal, abnormal):
        normal_labels = np.zeros((len(normal)))
        abnormal_labels = np.ones((len(abnormal)))
        train_x = np.expand_dims(normal[abnormal.shape[0]:],axis=-1)
        train_y = normal_labels[abnormal.shape[0]:]
        test_x = normal[:abnormal.shape[0]]
        test_y = normal_labels[:abnormal.shape[0]]
        test_x = np.expand_dims(np.concatenate((test_x, abnormal)),axis=-1)
        test_y = np.concatenate((test_y, abnormal_labels))

        return train_x, train_y, test_x, test_y


class MIMIIDatasetInterface():
    def __init__(self):
        self.ds = MIMIIDataset(spectogram_dir=_SPECTOGRAM_DATA_DIR, train_batch_size = 32, test_batch_size = 32)
        self.train_x, self.train_y = self.ds.train_dataset()
        self.test_x, self.test_y = self.ds.test_dataset()
        self.val_x, self.val_y = self.ds.val_dataset()

        self.train_batch_size = self.ds.train_batch_size
        self.test_batch_size = self.ds.test_batch_size

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

