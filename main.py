import glob
import os

import wavutils
import matplotlib.pyplot as plt
import models
from tqdm import tqdm
from sklearn import metrics
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

epochs = 10
n_mels=128
frames=5
n_fft=1024
hop_length=512
power=1.0
batch_size = 16
step = 3

base_directory = '/data/mimii_dataset'
train_features = "features/train.npy"
test_features = "features/test.npy"
dirs = sorted(glob.glob(os.path.abspath(f"{base_directory}/*/*/*")))

train_files, train_labels, eval_files, eval_labels = wavutils.dataset_generator(dirs[20])
if os.path.exists(train_features):
    X_train = wavutils.load_features(train_features)
else:
    X_train = wavutils.list_to_vector_array(train_files, n_mels=n_mels, frames=frames, n_fft=n_fft, hop_length=hop_length, power=power)
    wavutils.save_features(train_features, X_train)

if os.path.exists(test_features):
    X_test = wavutils.load_features(test_features)
else:
    X_test = wavutils.list_to_vector_array(eval_files, n_mels=n_mels, frames=frames, n_fft=n_fft, hop_length=hop_length, power=power)
    wavutils.save_features(test_features, X_test)

train_gen = wavutils.DataGenerator(train_features, batch_size=batch_size, dim=(32,128), shuffle=True, step=step)
test_gen = wavutils.DataGenerator(test_features, batch_size=batch_size, dim=(32,128), shuffle=False, step=step)
model = models.get_autoencoder_model()
# model = models.get_model((32,128),40)
model.summary()

history = model.fit_generator(train_gen,
                    validation_data=test_gen,
                    epochs=epochs,
                    verbose=1)

# model.save("the-model")

y_pred = [0. for k in eval_labels]
y_true = eval_labels


for file_idx, file_path in tqdm(enumerate(eval_files), total = len(eval_files)):
    vector_array = X_test[file_idx]
    length, _ = vector_array.shape

    dim = 32
    idex = np.arange(length - dim + step, step = step)

    for idx in range(len(idex)):
        start = min(idex[idx], length - dim)
        vector = vector_array[start:start+dim,:]

        vector = vector.reshape((1, vector.shape[0], vector.shape[1]))
        if idx == 0:
            batch = vector
        else:
            batch = np.concatenate((batch, vector))


    data = batch.reshape((batch.shape[0], batch.shape[1], batch.shape[2], 1))

    errors = np.mean(np.square(data - model.predict(data)), axis=-1)

    y_pred[file_idx] = np.mean(errors)
    

auc = metrics.roc_auc_score(y_true, y_pred)
p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
print(f"auc: {auc}, p_auc: {p_auc}")
plt.plot(y_pred)
plt.savefig("mse.png")
# for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
#     try:
#         data = np.expand_dims(wavutils.file_to_vector_array(file_name),axis=-1)
#         data = np.expand_dims(data,axis=0)
#         pred = model.predict(data)
#         error = np.mean(np.square(data - pred), axis=1)
#         y_pred[num] = np.mean(error)
#     except FileNotFoundError:
#         print("File broken!!: {}".format(file_name))

# score = metrics.roc_auc_score(y_true, y_pred)
# print(f"AUC: {score}")


