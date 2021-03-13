import glob
import os
from pathlib import Path

import wavutils
import matplotlib.pyplot as plt
import models
from tqdm import tqdm
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
import mimii_dataset
from sklearn.metrics import roc_curve

<<<<<<< HEAD
def as_numpy(t):
    return t.numpy()

tf.set_random_seed = 42
np.random.seed(42)
=======
>>>>>>> parent of 8c19d19... move to tf.keras
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

epochs = 10
n_mels=128
frames=5
n_fft=1024
hop_length=512
power=1.0
batch_size = 16
step = 3
scaling = False
base_directory = '/data/mimii_dataset'

dirs = sorted(glob.glob(os.path.abspath(f"{base_directory}/*/*/*")))[8:9]
fbaseline = open("baseline.txt","w")
for fpath in dirs:
    dataset = Path(fpath)
    dataset = dataset.parts
    sound_lvl = dataset[-3]
    machine_typ = dataset[-2]
    machine_id = dataset[-1]
    train_files, train_labels, eval_files, eval_labels = wavutils.dataset_generator(fpath)
    train_features = os.path.join("features",f"train_{sound_lvl}_{machine_typ}_{machine_id}.npy")
    test_features = os.path.join("features",f"test_{sound_lvl}_{machine_typ}_{machine_id}.npy")
    # test_features = os.path.join("features","test_6dB_fan_id_00.npy")
    if os.path.exists(train_features):
        ds = mimii_dataset.MIMIIDataset()
        x_train, _ = ds.get_train_dataset()
    else:
        print("data not found")

    if os.path.exists(test_features):
        X_test = wavutils.load_features(test_features)
        x_test, eval_labels = ds.get_test_dataset()
    else:
<<<<<<< HEAD
        print("data not found")

    model = tf_models.get_ds_autoencoder_model()
    model.summary()
    # model = tf_models.get_autoencoder_model()

    history = model.fit(x=x_train,
                        y=x_train,
                        epochs=epochs,
                        verbose=1)

    os.makedirs("saved_models", exist_ok=True) 
    tf.keras.models.save_model(model, "saved_models", save_format="tf")
=======
        X_test = wavutils.list_to_vector_array(eval_files, n_mels=n_mels, frames=frames, n_fft=n_fft, hop_length=hop_length, power=power)
        wavutils.save_features(test_features, X_test)

    if scaling:
        X_train = X_train / X_train.max()
        X_test = X_test / X_test.max()
        # from sklearn.preprocessing import Normalizer
        # norm = Normalizer()
        # # sc = StandardScaler()
        # nsamples, nx, ny = X_train.shape
        # X_train_flat = X_train.reshape((nsamples,nx*ny))
        # X_train_flat = norm.fit_transform(X_train_flat)
        # X_train = X_train_flat.reshape((nsamples, nx, ny))

        # nsamples, nx, ny = X_test.shape
        # X_test_flat = X_test.reshape((nsamples,nx*ny))
        # X_test_flat = norm.transform(X_test_flat)
        # X_test = X_test_flat.reshape((nsamples, nx, ny))

    train_gen = wavutils.DataGenerator(train_features, batch_size=batch_size, dim=(32,128), shuffle=True, step=step)
    test_gen = wavutils.DataGenerator(test_features, batch_size=batch_size, dim=(32,128), shuffle=False, step=step)
    model = models.get_autoencoder_model()
    model.summary()
    history = model.fit_generator(train_gen,
                        validation_data=test_gen,
                        epochs=epochs,
                        verbose=1)
>>>>>>> parent of 8c19d19... move to tf.keras

    y_pred = [0. for k in eval_labels]
    y_true = eval_labels
    recon = model.predict(x_test)
    recon = np.squeeze(recon)
    x_test = np.squeeze(x_test)

    errors = np.square(x_test - recon).mean(axis=2)
    y_pred = np.mean(errors,axis=1)

    plt.hist(y_pred, bins='auto')

<<<<<<< HEAD
    # fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # plt.plot(fpr,tpr)
    plt.savefig("mse_hist.png")
=======
            vector = vector.reshape((1, vector.shape[0], vector.shape[1]))
            if idx == 0:
                batch = vector
            else:
                batch = np.concatenate((batch, vector))


        data = batch.reshape((batch.shape[0], batch.shape[1], batch.shape[2], 1))

        errors = np.mean(np.square(data - model.predict(data)), axis=-1)

        y_pred[file_idx] = np.mean(errors)
        
>>>>>>> parent of 8c19d19... move to tf.keras

    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
    result_str = f"{sound_lvl},{machine_typ},{machine_id},{auc},{p_auc}\n"
    fbaseline.write(result_str)
    fbaseline.flush()
    print(result_str)

fbaseline.close()

