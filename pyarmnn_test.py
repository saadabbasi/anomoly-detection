import pyarmnn as ann
import numpy as np
import time

image = np.random.rand(1,128,32,1).astype(np.float32)

parser = ann.ITfLiteParser()
network = parser.CreateNetworkFromBinaryFile('test.tflite')

graph_id = 0
input_names = parser.GetSubgraphInputTensorNames(graph_id)
input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
input_tensor_id = input_binding_info[0]
input_tensor_info = input_binding_info[1]
print('tensor id: ' + str(input_tensor_id))
print('tensor info: ' + str(input_tensor_info))
# Create a runtime object that will perform inference.
options = ann.CreationOptions()
runtime = ann.IRuntime(options)

preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

# Load the optimized network into the runtime.
net_id, _ = runtime.LoadNetwork(opt_network)
print("Loaded network, id={net_id}")
# Create an inputTensor for inference.
input_tensors = ann.make_input_tensors([input_binding_info], [image])

# Get output binding information for an output layer by using the layer name.
output_names = parser.GetSubgraphOutputTensorNames(graph_id)
output_binding_info = parser.GetNetworkOutputBindingInfo(0, output_names[0])
output_tensors = ann.make_output_tensors([output_binding_info])

bench = np.zeros((100))
for n in range(100):
    start_t = time.perf_counter_ns()
    runtime.EnqueueWorkload(0, input_tensors, output_tensors)
    end_t = time.perf_counter_ns()
    bench[n] = end_t - start_t

print(f"Average time: {np.min(bench)/1e6} us")
results = ann.workload_tensors_to_ndarray(output_tensors)