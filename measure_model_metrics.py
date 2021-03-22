import os
os.system("source /opt/intel/openvino_2021/bin/setupvars.sh")
import tensorflow as tf
import numpy as np
import os
import mimii_dataset
from time import perf_counter_ns
import openvino
from openvino.inference_engine import IENetwork, IECore
import argparse

def convert_model_to_IR(frozen_model_path, output_name):
    frozen_model_path = frozen_model_path.replace(" ","\ ")
    output_dir = os.path.dirname(frozen_model_path)
    cmd_string = f"python /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_tf.py --input_model {frozen_model_path} --input \"input_1[1 32 128 1]\" -n {output_name} -o {output_dir}"
    os.system(cmd_string)
    # python /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_tf.py --input_model frozen.pb --input "input_1[1 32 128 1]"

def benchmark_tf_model_ns(sess, OUTPUT_TENSOR, feed_dict, N = 100):
    time_t = np.zeros((N,))
    for n in range(N):
        start_t = perf_counter_ns()
        out = sess.run([OUTPUT_TENSOR], feed_dict = feed_dict)
        end_t = perf_counter_ns()
        time_t[n] = end_t - start_t

    return time_t.min()

def benchmark_openvino_model_ns(model, shape, N = 100):
    etime = np.zeros((N,))
    if type(model) == openvino.inference_engine.ie_api.ExecutableNetwork:
        x = np.random.randn(shape[0],shape[1],shape[2],shape[3])
        for i in range(100):
            start_t = perf_counter_ns()
            model.infer({'input_1':x})
            end_t = perf_counter_ns()
            etime[i] = end_t - start_t
    
    return etime.min()

def load_openvino_model(xml_path):
    bin_path = os.path.splitext(xml_path)[0] + ".bin"
    ie = IECore()
    net = ie.read_network(model=xml_path, weights = bin_path)
    exec_net = ie.load_network(network = net, device_name='CPU', num_requests=0)
    return exec_net

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

def decode_model_path(model_dir):
    gensynth_trial_id = os.path.split(model_dir)[1].split("-")[0]
    return gensynth_trial_id

def measure_latency_ns(model_dir):
    gensynth_trial_id = decode_model_path(model_dir)
    macro_arch = gensynth_trial_id.split("_")[1]

    if macro_arch == 'SC':
        IMAGE_INPUT_TENSOR = "input_1:0"
        TARGET_INPUT_TENSOR = "conv2d_6_target:0"
        OUTPUT_TENSOR = "conv2d_6/BiasAdd:0"
        LOSS_TENSOR = "loss/mul:0"
    elif macro_arch == 'DW':
        IMAGE_INPUT_TENSOR = "input_1:0"
        TARGET_INPUT_TENSOR = "separable_conv2d_6_target:0"
        OUTPUT_TENSOR = "separable_conv2d_6/BiasAdd:0"
        LOSS_TENSOR = "loss/mul:0"
    else:
        raise ValueError(f"{macro_arch} not recognized. Is your directory structure and name OK?")

    
    meta_path = os.path.join(model_dir, "model.meta")
    ckpt_path = model_dir
    frozen_path = os.path.join(model_dir, "frozen.pb")
    openvino_output_name = "vino"
    openvino_path = os.path.join(model_dir, f"{openvino_output_name}.xml")

    convert_model_to_IR(frozen_path,openvino_output_name)
    model = load_openvino_model(openvino_path)
    openvino_min = benchmark_openvino_model_ns(model, (1,1,32,128))

    feed_dict = {IMAGE_INPUT_TENSOR: np.random.rand(1,32,128,1)}
    N = 100
    run_meta = tf.RunMetadata()
    graph,sess,saver = load_graph(meta_path)

    with sess.as_default():
        load_ckpt(sess, saver, ckpt_path)
        tf_min = benchmark_tf_model_ns(sess, OUTPUT_TENSOR, feed_dict, N = 10)

    return {'openvino':openvino_min, 'tensorflow':tf_min}


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    args = parser.parse_args()
    
    model_dir = args.model_dir

    models = [os.path.join(model_dir,name) for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir,name))]
    with open("results.txt","w") as f:
        for dir in models:
            latency = measure_latency_ns(dir)
            gensynth_trial_id = decode_model_path(dir)
            f.write(f"{gensynth_trial_id},{latency['openvino']/1e6},{latency['tensorflow']/1e6}\n")
                
                
