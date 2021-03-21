import tensorflow as tf
import numpy as np
import os
import mimii_dataset
from sklearn import metrics
import matplotlib.pyplot as plt

IMAGE_INPUT_TENSOR = "input_1:0"
TARGET_INPUT_TENSOR = "separable_conv2d_6_target:0"
OUTPUT_TENSOR = "separable_conv2d_6/BiasAdd:0"
LOSS_TENSOR = "loss/mul:0"

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
    ax1.hist(y_pred[np.where(y_true==0)], label="Normal", bins=100)
    ax1.hist(y_pred[np.where(y_true==1)], label="Abnormal", alpha=alpha, bins=100)
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

def load_graph(meta_graph_file):
    graph = tf.Graph()
    with graph.as_default():
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=sess_config)
        saver = tf.train.import_meta_graph(meta_graph_file)
    return graph, sess, saver

def load_ckpt(sess, saver, ckpt_file):
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_file))

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model_path = "/home/saad.abbasi/code/anomoly-detection/gs_models/TA_DW_4_0dB_fan_id_04/model.meta"
ckpt_path = "/home/saad.abbasi/code/anomoly-detection/gs_models/TA_DW_4_0dB_fan_id_04/"
device_to_run = '7'

ds = mimii_dataset.MIMIIDataset('0dB_fan_id_04', format = '3D')

test_x, test_y = ds.test_dataset()

graph, sess, saver = load_graph(model_path)
feed_dict = {IMAGE_INPUT_TENSOR: test_x}
with graph.as_default():
    load_ckpt(sess, saver, ckpt_path)

    sess.graph.get_tensor_by_name(LOSS_TENSOR)
    out = sess.run([OUTPUT_TENSOR], feed_dict = feed_dict)

auc = compute_AUC(out, test_x, test_y)
y_pred = compute_probs(out, test_x)
print(f"auc {auc:.3f}")
plot_distribution("GS_hist.png",test_y,y_pred)