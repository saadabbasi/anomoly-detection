import tensorflow as tf
import numpy as np
import os
import mimii_dataset
from time import perf_counter_ns

def load_frozen_graph(frozen_graph_filename):
    import tensorflow as tf
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def load_graph(meta_graph_file):
    graph = tf.Graph()
    with graph.as_default():
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=sess_config)
        saver = tf.train.import_meta_graph(meta_graph_file)
    return graph, sess, saver

def load_ckpt(sess, saver, ckpt_file):
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_file))

def get_flops(sess, run_meta):
    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))

def tf_network_latency(sess, OUTPUT_TENSOR, feed_dict, N = 100):
    time_t = np.zeros((N,))
    for n in range(N):
        start_t = perf_counter_ns()
        out = sess.run([OUTPUT_TENSOR], feed_dict = feed_dict)
        end_t = perf_counter_ns()
        time_t[n] = end_t - start_t

    return time_t.min()

# IMAGE_INPUT_TENSOR = "input_1:0"
# TARGET_INPUT_TENSOR = "separable_conv2d_6_target:0"
# OUTPUT_TENSOR = "separable_conv2d_6/BiasAdd:0"
# LOSS_TENSOR = "loss/mul:0"

IMAGE_INPUT_TENSOR = "input_1:0"
TARGET_INPUT_TENSOR = "conv2d_6_target:0"
OUTPUT_TENSOR = "conv2d_6/BiasAdd:0"
LOSS_TENSOR = "loss/mul:0"

model_dir = "/home/saad.abbasi/code/anomoly-detection/model_archives/TA_SC_6dB_fan_id_00-1670_4-2021-03-21_22 30 37/"
model_dir = "/home/saad.abbasi/code/anomoly-detection/model_archives/TA_SC_6dB_fan_id_02-1673_10-2021-03-21_22 51 59/"
model_dir = "/home/saad.abbasi/code/anomoly-detection/model_archives/TA_SC_6dB_fan_id_00-1556_4-2021-03-21_00 11 37"
meta_path = os.path.join(model_dir, "model.meta")
ckpt_path = model_dir
device_to_run = ""
os.environ["CUDA_VISIBLE_DEVICES"] = device_to_run

ds = mimii_dataset.MIMIIDataset('min6dB_slider_id_00', format = '3D', train_batch_size=1, test_batch_size=1)

test_x, test_y = ds.test_dataset()

# feed_dict = {IMAGE_INPUT_TENSOR: test_x}
feed_dict = {IMAGE_INPUT_TENSOR: np.random.rand(1,32,128,1)}
N = 100
run_meta = tf.RunMetadata()
graph,sess,saver = load_graph(meta_path)

with sess.as_default():
    load_ckpt(sess, saver, ckpt_path)
    min = tf_network_latency(sess, OUTPUT_TENSOR, feed_dict, N = 10) / 1e6
    print(f"Min: {min:.3f} µs")