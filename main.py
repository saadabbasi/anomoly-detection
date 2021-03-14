import glob
import os
from pathlib import Path

import tensorflow as tf
import wavutils
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
import mimii_dataset
from sklearn.metrics import roc_curve

def compute_probs(output_imgs, target_imgs):
    # should I be calling these imgs?
    recon = np.squeeze(output_imgs)
    test_x = np.squeeze(target_imgs)
    errors = np.square(test_x - recon).mean(axis=2)
    y_pred = np.mean(errors,axis=1)
    return y_pred

def plot_distribution(fname, y_true, y_pred):
    alpha = 0.75
    fig, (ax1, ax2) = plt.subplots(ncols=2,nrows=1)
    ax1.hist(y_pred[np.where(y_true==0)], label="Normal")
    ax1.hist(y_pred[np.where(y_true==1)], label="Abnormal", alpha=alpha)
    ax2.plot(y_pred[np.where(y_true==0)], label="Normal")
    ax2.plot(y_pred[np.where(y_true==1)], label="Abnormal", alpha=alpha)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)

def compute_AUC(output_imgs, target_imgs, y_true, plot_hist=False):
    y_pred = compute_probs(output_imgs, target_imgs)
    auc = metrics.roc_auc_score(y_true, y_pred)
    return auc

import tf_models

tf.set_random_seed = 42
np.random.seed(42)
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
base_directory = "/home/saad.abbasi/datasets/mimii"

dirs = sorted(glob.glob(os.path.abspath(f"{base_directory}/*/*/*")))
fbaseline = open("baseline.txt","w")

ds = mimii_dataset.MIMIIDataset(machine_type_id = "6dB_fan_id_06")
train_x, _ = ds.train_dataset()
test_x, eval_labels = ds.test_dataset()

model = tf_models.get_ds_autoencoder_model(use_batchnorm=False, h_space_flat=False)
# tf.keras.models.save_model(model, "tiny_anomoly_ds", save_format="h5")
preds = model.predict(test_x)
auc_prior = compute_AUC(preds, test_x, eval_labels, plot_hist=True)
y_pred = compute_probs(preds, test_x)
plot_distribution("hist-prior.png", eval_labels, y_pred)
model.summary()

history = model.fit(x=train_x, y=train_x, epochs=epochs,verbose=1)

preds = model.predict(test_x)
y_pred = compute_probs(preds, test_x)
auc_after = compute_AUC(preds, test_x, eval_labels, plot_hist=True)

plot_distribution("hist.png", eval_labels, y_pred)

print(f"AUC Before Training: {auc_prior:.3f} After: {auc_after:.3f}")
os.makedirs("saved_models", exist_ok=True) 
# print([n.name for n in tf.get_default_graph().as_graph_def().node])