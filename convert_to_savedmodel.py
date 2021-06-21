import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

def load_frozen_graph(model_filepath):
    with tf.io.gfile.GFile(model_filepath, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

model_filepath = "/home/saad/model_archives/edge_build/TA_D_SC_0dB_slider_id_00-1938_9-2021-03-26_03 28 48/frozen.pb"
export_dir = "test"
graph_def = load_frozen_graph(model_filepath)
builder = tf.compat.v1.saved_model.Builder(export_dir)
sigs = {}

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    g = tf.compat.v1.get_default_graph()
    inp = g.get_tensor_by_name("input_1:0")
    out = g.get_tensor_by_name("conv2d_10/BiasAdd:0")

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
            {"in": inp}, {"out": out})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

builder.save()