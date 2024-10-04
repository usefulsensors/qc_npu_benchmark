# Benchmark script for the Qualcomm NPU on a Microsoft Surface Pro Tablet.
# See README.md for more information, and LICENSE for copyright information.

import numpy as np
import onnx
from onnx import helper as h, TensorProto as tp
import onnxruntime as ort
from onnxruntime.quantization import QuantFormat, QuantType, CalibrationDataReader, quantize_static
from readable_number import ReadableNumber
import time

# Define the shape of the matrix multiplication operation to benchmark.
MATRIX_COUNT = 6
MATRIX_A = 1500
MATRIX_B = 1500
MATRIX_K = 256
INPUT0_SHAPE = [1, MATRIX_COUNT, MATRIX_A, MATRIX_K]
INPUT1_SHAPE = [1, MATRIX_COUNT, MATRIX_K, MATRIX_B]
OUTPUT_SHAPE = [1, MATRIX_COUNT, MATRIX_A, MATRIX_B]

# A multiply-add counts as two operations, conventionally.
OPS_PER_MUL_ADD = 2

# Derive the total number of operations from the input shapes.
OPS_PER_INFERENCE = OPS_PER_MUL_ADD * MATRIX_COUNT * MATRIX_A * MATRIX_B * MATRIX_K

# The float range to distribute random inputs over.
INPUT_RANGE = 1.0 / 5.0

# Where to save the intermediate model files. These will overwrite whatever is
# in the existing repository by default.
FLOAT_MODEL_PATH = "matmul_model_float.onnx"
QUANT_MODEL_PATH = "matmul_model_quant.onnx"
QUANT_IO_MODEL_PATH = "matmul_model_quant_io.onnx"

# How many times to run inference on the model, to obtain the mean latency.
ITERATIONS = 20

# This class is used to provide calibration inputs for the quantization 
# process. Since we only care about accuracy for the two inputs we set up, we
# just return those examples and then signal that we're done.
class MockDataReader(CalibrationDataReader):
    def __init__(self, input0, input1):
        self.has_run = False
        self.input0 = input0
        self.input1 = input1

    def get_next(self):
        if self.has_run:
            return None
        else:
            self.has_run = True
            return {
                "input0_tensor": self.input0,
                "input1_tensor": self.input1,
            }

    def rewind(self):
        self.has_run = False

# Convert a float tensor into an eight-bit equivalent.
def quantize_tensor(input, scale, zero_point):
    assert input.dtype == np.float32
    return np.clip(np.round(input / scale) + zero_point, 0, 255).astype(np.uint8)

# Convert an eight-bit quantized tensor into a float result.
def dequantize_tensor(input, scale, zero_point):
    assert input.dtype == np.uint8
    return (input.astype(np.float32) - zero_point) * scale

# Use the Onnx model framework to construct a graph containing a single matrix
# multiplication operation with two dynamic inputs, all with float computation.
def make_matmul_float_model():

    matmul_node = h.make_node(
        "MatMul", 
        inputs=["input0_tensor", "input1_tensor"],
        outputs=["matmul_output_tensor"], 
        name="matmul_node")
    
    matmul_float_graph = h.make_graph(
        nodes=[
            matmul_node,
        ], 
        name="matmul_float_graph",
        inputs=[
            h.make_tensor_value_info("input0_tensor", tp.FLOAT, INPUT0_SHAPE),
            h.make_tensor_value_info("input1_tensor", tp.FLOAT, INPUT1_SHAPE)
        ],
        outputs=[
            h.make_tensor_value_info("matmul_output_tensor", tp.FLOAT, OUTPUT_SHAPE),
        ],
        initializer=[
        ])

    matmul_float_model = h.make_model(matmul_float_graph, producer_name="matmul_test")

    return matmul_float_model

# Create the base float model and save it out.
matmul_float_model = make_matmul_float_model()
onnx.checker.check_model(matmul_float_model)
onnx.save(matmul_float_model, FLOAT_MODEL_PATH)

# Arbitrary but fixed seed value.
rng = np.random.default_rng(7528840384)

# We generate two input tensors with random values from zero to INPUT_RANGE.
input0_numpy = rng.random((INPUT0_SHAPE)).astype(np.float32) * INPUT_RANGE
input1_numpy = rng.random((INPUT1_SHAPE)).astype(np.float32) * INPUT_RANGE

matmul_output_numpy = np.zeros(OUTPUT_SHAPE, dtype=np.float32)

input0_tensor = ort.OrtValue.ortvalue_from_numpy(input0_numpy, 'cuda', 0)
input1_tensor = ort.OrtValue.ortvalue_from_numpy(input1_numpy, 'cuda', 0)
matmul_output_tensor = ort.OrtValue.ortvalue_from_numpy(matmul_output_numpy, 'cuda', 0)

# Create an Onnx Runtime session to run the model on the CPU.
gpu_options = ort.SessionOptions()
gpu_session = ort.InferenceSession(
    FLOAT_MODEL_PATH,
    sess_options=gpu_options,
    providers=[("CUDAExecutionProvider", {"enable_cuda_graph": '1'})],
)

io_binding = gpu_session.io_binding()

# Pass gpu_graph_id to RunOptions through RunConfigs
ro = ort.RunOptions()
# gpu_graph_id is optional if the session uses only one cuda graph
ro.add_run_config_entry("gpu_graph_id", "1")

# Bind the input and output
io_binding.bind_ortvalue_input("input0_tensor", input0_tensor)
io_binding.bind_ortvalue_input("input1_tensor", input1_tensor)
io_binding.bind_ortvalue_output("matmul_output_tensor", matmul_output_tensor)

# Run the float model multiple times on the CPU, and calculate the overall latency.
for i in range(ITERATIONS + 1):
    # Skip the first run, since there's setup and caching. Sae
    # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
    if i == 1:
        start_gpu = time.time()
    input0_tensor.update_inplace(input0_numpy)
    input1_tensor.update_inplace(input1_numpy)
    gpu_session.run_with_iobinding(io_binding, ro)
end_gpu = time.time()
# print(matmul_output_tensor.numpy())

print("************ Benchmark Results ************")

gpu_s = (end_gpu - start_gpu) / ITERATIONS

gpu_ms = gpu_s * 1000.0

# Derive the ops per second from the latency and number of ops in the model.
gpu_ops_per_second = round(OPS_PER_INFERENCE / gpu_s)

rn = ReadableNumber(precision=0, digit_group_size=3)

print(f"GPU took {gpu_ms:0.2f}ms, {rn.of(gpu_ops_per_second)} ops per second")
