from operator import delitem
import os
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import mimii_dataset
import tf_models

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
    ax1.set_title("MSE Histrogram")
    ax2.plot(y_pred[np.where(y_true==0)], label="Normal")
    ax2.plot(y_pred[np.where(y_true==1)], label="Abnormal", alpha=alpha)
    ax2.set_title("Raw MSE Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)

def compute_AUC(output_imgs, target_imgs, y_true, plot_hist=False):
    y_pred = compute_probs(output_imgs, target_imgs)
    auc = metrics.roc_auc_score(y_true, y_pred)
    return auc


tf.set_random_seed = 42
np.random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

epochs = 100
n_mels=128
frames=5
n_fft=1024
hop_length=512
power=1.0
batch_size = 16
step = 3
scaling = False

spectrograms_fnames = ['0dB_fan_id_00','0dB_fan_id_02','0dB_fan_id_04','0dB_fan_id_06',
                        'min6dB_fan_id_00', 'min6dB_fan_id_02', 'min6dB_fan_id_04', 'min6dB_fan_id_06',
                        '6dB_fan_id_00', '6dB_fan_id_02', '6dB_fan_id_04','6dB_fan_id_06',
                        '0dB_slider_id_00','0dB_slider_id_02','0dB_slider_id_04','0dB_slider_id_06',
                        'min6dB_slider_id_00','min6dB_slider_id_02','min6dB_slider_id_04','min6dB_slider_id_06',
                        '6dB_slider_id_00','6dB_slider_id_02','6dB_slider_id_04','6dB_slider_id_06']

fbaseline = open("baseline.txt","w")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for machine in spectrograms_fnames:
    ds = mimii_dataset.MIMIIDataset(machine_type_id = machine)
    train_x, _ = ds.train_dataset()
    test_x, eval_labels = ds.test_dataset()
    raw = np.mean(test_x,axis=-1).mean(axis=-1).mean(axis=-1)

    plot_distribution("hist-raw.png", eval_labels, raw)

    # model = tf_models.get_autoencoder_model_s(tf_models.standard_conv(), use_batchnorm=True)
    model = tf_models.conv_baseline()

    preds = model.predict(test_x)
    auc_prior = compute_AUC(preds, test_x, eval_labels, plot_hist=True)
    y_pred = compute_probs(preds, test_x)
    plot_distribution("hist-prior.png", eval_labels, y_pred)


    history = model.fit(x=train_x,y=train_x, epochs=epochs,verbose=1)
    preds = model.predict(test_x)
    y_pred = compute_probs(preds, test_x)
    auc_after = compute_AUC(preds, test_x, eval_labels, plot_hist=True)
    plot_distribution(f"{machine}_hist.png", eval_labels, y_pred)

    sound_lvl, machine_typ, _, machine_id = machine.split('_')
    result_str = f"{sound_lvl},{machine_typ},{machine_id},{auc_after},{auc_prior}\n"
    print(result_str)
    fbaseline.write(result_str)
    fbaseline.flush()

fbaseline.close()
# os.makedirs("saved_models", exist_ok=True) 
# print([n.name for n in tf.get_default_graph().as_graph_def().node])