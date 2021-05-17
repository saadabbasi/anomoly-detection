import os
import tensorflow as tf
import numpy as np
import os
from time import perf_counter_ns
import argparse

def benchmark_tf_model_ns(sess, OUTPUT_TENSOR, feed_dict, N = 100):
    time_t = np.zeros((N,))
    for n in range(N):
        start_t = perf_counter_ns()
        out = sess.run([OUTPUT_TENSOR], feed_dict = feed_dict)
        end_t = perf_counter_ns()
        time_t[n] = end_t - start_t

    return time_t.min()


def load_frozen_graph(frozen_graph_filename):
    import tensorflow as tf
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=sess_config)
    return graph, sess

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

def decode_model_path(model_dir):
    gensynth_trial_id = os.path.split(model_dir)[1].split("-")[0]
    return gensynth_trial_id

def measure_latency_ns(model_dir, n_iterations = 10):
    gensynth_trial_id = decode_model_path(model_dir)
    macro_arch = gensynth_trial_id.split("_")[1]
    print(gensynth_trial_id,macro_arch)

    IMAGE_INPUT_TENSOR = "input_1:0"
    LOSS_TENSOR = "loss/mul:0"
    if macro_arch == 'SC':
        TARGET_INPUT_TENSOR = "conv2d_6_target:0"
        OUTPUT_TENSOR = "conv2d_6/BiasAdd:0"
    elif macro_arch == 'B':
        TARGET_INPUT_TENSOR = "conv2d_transpose_4_target:0"
        OUTPUT_TENSOR = "conv2d_transpose_4/BiasAdd:0"
    elif macro_arch == 'D':
        TARGET_INPUT_TENSOR = "conv2d_10_target:0"
        OUTPUT_TENSOR = "conv2d_10/BiasAdd:0"
    elif macro_arch == 'DW':
        TARGET_INPUT_TENSOR = "separable_conv2d_6_target:0"
        OUTPUT_TENSOR = "separable_conv2d_6/BiasAdd:0"
    else:
        raise ValueError(f"{macro_arch} not recognized. Is your directory structure and name OK?")

    
    meta_path = os.path.join(model_dir, "model.meta")
    ckpt_path = model_dir
    frozen_path = os.path.join(model_dir, "frozen.pb")

    feed_dict = {IMAGE_INPUT_TENSOR: np.random.rand(1,32,128,1)}
    run_meta = tf.RunMetadata()
    graph,sess,saver = load_graph(meta_path)

    with sess.as_default():
        load_ckpt(sess, saver, ckpt_path)
        tf_min = benchmark_tf_model_ns(sess, OUTPUT_TENSOR, feed_dict, N = n_iterations)

    return {'tensorflow':tf_min}


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mdir', action='store', dest="model_dir")
    group.add_argument('--baseline', action='store', dest="model_path")
    args = parser.parse_args()

    if args.model_dir:
        print(f"Loading all models from {args.model_dir}")
        models = [os.path.join(args.model_dir,name) for name in os.listdir(args.model_dir) if os.path.isdir(os.path.join(args.model_dir,name))]
        print(models)
        with open("results.txt","w") as f:
            f.write("Trial ID, openvino latency (us), tensorflow latency (us)\n")
            for dir in models:
                print(f"Measuring model in {dir}")
                latency = measure_latency_ns(dir,n_iterations = 100)
                gensynth_trial_id = decode_model_path(dir)
                f.write(f"{gensynth_trial_id},{latency['tensorflow']/1e6}\n")
    else:
        print(f"Loading baseline model from: {args.model_path}")
        IMAGE_INPUT_TENSOR = "prefix/input_1:0"
        OUTPUT_TENSOR = "prefix/conv2d_transpose_4/BiasAdd:0"

        feed_dict = {IMAGE_INPUT_TENSOR: np.random.rand(1,32,128,1)}
        graph, sess = load_frozen_graph(args.model_path)

        # print([n.name for n in graph.as_graph_def().node])

        tf_t = benchmark_tf_model_ns(sess, OUTPUT_TENSOR, feed_dict)

        print("========RESULTS=========")
        print(f"Tensorflow RPI: {tf_t/1e6} µs")
        print("========================")
    
