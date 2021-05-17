import tensorflow as tf
import numpy as np
from tf_models import conv_baseline


def load_ckpt(sess, saver, ckpt_file):
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_file))

def load_graph(meta_graph_file):
    graph = tf.Graph()
    with graph.as_default():
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=sess_config)
        saver = tf.train.import_meta_graph(meta_graph_file)
    return graph, sess, saver

# IMAGE_INPUT_TENSOR = "input_1:0"
# OUTPUT_TENSOR = "conv2d_10/BiasAdd:0"
# feed_dict = {IMAGE_INPUT_TENSOR: np.random.rand(1,32,128,1)}
# run_meta = tf.RunMetadata()
# # meta_path = os.path.join(model_dir, "model.meta")
# meta_path = "/home/pi/model_archives/edge_build/TA_D_SC_0dB_slider_id_00-1938_9-2021-03-26_03 28 48/model.meta"
# ckpt_path = "/home/pi/model_archives/edge_build/TA_D_SC_0dB_slider_id_00-1938_9-2021-03-26_03 28 48/"
# tf.compat.v1.enable_control_flow_v2()

# with tf.Graph().as_default():
#     graph, sess, saver = load_graph(meta_path)
#     tf.train.write_graph(sess.graph_def, logdir=".", name='test.pb')

# converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
#     graph_def_file = "test.pb",
#     input_arrays = ['input_1'],
#     input_shapes = {'input_1' : [1,32,128,1]},
#     output_arrays = ['conv2d_10/BiasAdd']
# )

model = conv_baseline()
tf.keras.models.save_model(model, "test.h5")

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("test.h5")
tflite_model = converter.convert()

with open("test.tflite", "wb") as f:
    f.write(tflite_model)