import tensorflow as tf
import os
import numpy as np
import copy
from tensorflow.core.framework import graph_pb2

def load_meta_graph(meta_graph_file):
    graph = tf.Graph()
    with graph.as_default():
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=sess_config)
        saver = tf.train.import_meta_graph(meta_graph_file)
    return graph, sess, saver

def replace_node(model_path, node_name, shape):
    example_graph = tf.Graph()
    # input_const = np.ones(shape)
    with tf.Session(graph = example_graph):
        # c = tf.constant(input_const, dtype=tf.float32, shape = shape, name = node_name)
        c = tf.placeholder(dtype = tf.float32, shape = shape, name = node_name)
        for node in example_graph.as_graph_def().node:
            if node.name == node_name:
                c_def = node

    old_graph, _, _ = load_meta_graph(model_path)
    old_graph_def = old_graph.as_graph_def()
    new_graph_def = graph_pb2.GraphDef()
    for node in old_graph_def.node:
        if node.name == 'input_1':
            new_graph_def.node.extend([c_def])
        else:
            new_graph_def.node.extend([copy.deepcopy(node)])

    return new_graph_def

def save_model(save_path, graph_def):
    with tf.gfile.GFile(save_path,"wb") as f:
        f.write(graph_def.SerializeToString())

if __name__ == "__main__":
    model_dir = "/home/saad.abbasi/code/anomoly-detection/model_archives/TA_SC_6dB_fan_id_00-1556_4-2021-03-21_00 11 37"
    model_dir = "/home/saad.abbasi/code/anomoly-detection/model_archives/TA_SC_6dB_fan_id_06-1672_12-2021-03-21_22 52 13"
    meta_path = os.path.join(model_dir, "model.meta")
    device_to_run = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = device_to_run

    shape = (1,32,128,1)
    node_name = "input_1"
    graph_def = replace_node(meta_path, node_name, shape)
    save_model("test_model.pb",graph_def)

