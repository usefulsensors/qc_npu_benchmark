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

# Builds a model with a single matrix multiplication operation, computing all
# values in eight-bit, with eight-bit inputs and outputs.
def make_matmul_quantized_io_model(
    input_scale, input_zero_point, 
    matmul_scale, matmul_zero_point):
    input0_scale_tensor = h.make_tensor(
        name="input0_scale", 
        data_type=tp.FLOAT, 
        dims=[1],
        vals=[input_scale])

    input0_zero_point_tensor = h.make_tensor(
        name="input0_zero_point", 
        data_type=tp.UINT8, 
        dims=[1],
        vals=[input_zero_point])

    input0_dequant_node = h.make_node(
        "DequantizeLinear", 
        inputs=["input0_quant_tensor", "input0_scale", "input0_zero_point"], 
        outputs=["input0_dequant_tensor"],
        name="input0_dequant")

    input1_scale_tensor = h.make_tensor(
        name="input1_scale", 
        data_type=tp.FLOAT, 
        dims=[1],
        vals=[input_scale])

    input1_zero_point_tensor = h.make_tensor(
        name="input1_zero_point", 
        data_type=tp.UINT8, 
        dims=[1],
        vals=[input_zero_point])

    input1_dequant_node = h.make_node(
        "DequantizeLinear", 
        inputs=["input1_quant_tensor", "input1_scale", "input1_zero_point"], 
        outputs=["input1_dequant_tensor"],
        name="input1_dequant")

    matmul_node = h.make_node(
        "MatMul", 
        inputs=["input0_dequant_tensor", "input1_dequant_tensor"],
        outputs=["matmul_output_tensor"], 
        name="matmul_node")

    matmul_output_scale_tensor = h.make_tensor(
        name="matmul_output_scale", 
        data_type=tp.FLOAT, 
        dims=[1],
        vals=[matmul_scale])

    matmul_output_zero_point_tensor = h.make_tensor(
        name="matmul_output_zero_point", 
        data_type=tp.UINT8, 
        dims=[1],
        vals=[matmul_zero_point])

    matmul_output_quant_node = h.make_node(
        "QuantizeLinear", 
        inputs=["matmul_output_tensor", "matmul_output_scale", "matmul_output_zero_point"], 
        outputs=["matmul_output_quant_tensor"],
        name="matmul_output_quant")

    matmul_output_dequant_node = h.make_node(
        "DequantizeLinear", 
        inputs=["matmul_output_quant_tensor", "matmul_output_scale", "matmul_output_zero_point"], 
        outputs=["matmul_output_dequant_tensor"],
        name="matmul_output_dequant")

    matmul_quantized_graph = h.make_graph(
        nodes=[
            input0_dequant_node,
            input1_dequant_node,
            matmul_node,
            matmul_output_quant_node,
            matmul_output_dequant_node,
        ], 
        name="matmul_quantized_graph",
        inputs=[
            h.make_tensor_value_info("input0_quant_tensor", tp.UINT8, INPUT0_SHAPE),
            h.make_tensor_value_info("input1_quant_tensor", tp.UINT8, INPUT1_SHAPE)
        ],
        outputs=[
            h.make_tensor_value_info("matmul_output_quant_tensor", tp.UINT8, OUTPUT_SHAPE),
        ],
        initializer=[
            input0_scale_tensor,
            input0_zero_point_tensor,
            input1_scale_tensor,
            input1_zero_point_tensor,
            matmul_output_scale_tensor,
            matmul_output_zero_point_tensor,
        ])

    matmul_quantized_model = h.make_model(matmul_quantized_graph, producer_name="matmul_test")

    return matmul_quantized_model

# Calculates the mean difference between two arrays.
def array_msd(x, y):
    difference = x - y
    msd = np.mean(np.sqrt(difference * difference))
    return msd

# Create the base float model and save it out.
matmul_float_model = make_matmul_float_model()
onnx.checker.check_model(matmul_float_model)
onnx.save(matmul_float_model, FLOAT_MODEL_PATH)

# Arbitrary but fixed seed value.
rng = np.random.default_rng(7528840384)

# We generate two input tensors with random values from zero to INPUT_RANGE.
input0_tensor = rng.random((INPUT0_SHAPE)).astype(np.float32) * INPUT_RANGE
input1_tensor = rng.random((INPUT1_SHAPE)).astype(np.float32) * INPUT_RANGE

# Create an Onnx Runtime session to run the model on the CPU.
cpu_options = ort.SessionOptions()
cpu_session = ort.InferenceSession(
    FLOAT_MODEL_PATH,
    sess_options=cpu_options,
)

# Run the float model multiple times on the CPU, and calculate the overall latency.
start_cpu = time.time()
for i in range(ITERATIONS):
    cpu_float_outputs = cpu_session.run(
        None, {
            "input0_tensor": input0_tensor,
            "input1_tensor": input1_tensor,
        })
end_cpu = time.time()
cpu_float_output = cpu_float_outputs[0]

# Create a quantized model using the recommended ORT method. This has float
# inputs and outputs.
data_reader = MockDataReader(input0_tensor, input1_tensor)
quantize_static(
    FLOAT_MODEL_PATH,
    QUANT_MODEL_PATH,
    data_reader,
    quant_format=QuantFormat.QDQ,
    per_channel=False,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
)

# Create an ORT session that should run on the NPU.
npu_quant_options = ort.SessionOptions()
# Raise an error if any operations aren't runnable on the NPU.
npu_quant_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
npu_quant_session = ort.InferenceSession(
    QUANT_MODEL_PATH,
    sess_options=npu_quant_options,
    providers=["QNNExecutionProvider"],
    provider_options=[{
        "backend_path": "QnnHtp.dll",
        "htp_performance_mode": "sustained_high_performance",
        "enable_htp_fp16_precision": "1",
        # "profiling_level": "detailed",
        # "profiling_file_path": "npu_quant_profile.csv",
    }]
)

# Run the quantized model with float I/O on the NPU and calculate the latency.
start_npu_quant = time.time()
for i in range(ITERATIONS):
    npu_quant_outputs = npu_quant_session.run(
        None, {
            "input0_tensor": input0_tensor,
            "input1_tensor": input1_tensor,
        })
end_npu_quant = time.time()
npu_quant_output = npu_quant_outputs[0]

# Build a quantized model that has quantized inputs and outputs, to avoid the
# performance problems we've seen with float conversion. We can't use the
# standard quantize_static method, so instead construct the model from scratch.
input_scale = INPUT_RANGE / 255.0
input_zero_point = 0
max_output = np.max(cpu_float_output)
matmul_scale = max_output / 255.0
matmul_zero_point = 0
quant_io_model =  make_matmul_quantized_io_model(input_scale, input_zero_point, matmul_scale, matmul_zero_point)
onnx.checker.check_model(quant_io_model)
onnx.save(quant_io_model, QUANT_IO_MODEL_PATH)

# Convert our float inputs into quantized equivalents.
input0_quant_tensor = quantize_tensor(input0_tensor, input_scale, input_zero_point)
input1_quant_tensor = quantize_tensor(input1_tensor, input_scale, input_zero_point)

# Build an NPU session to run the fully-quantized model.
npu_quant_io_options = ort.SessionOptions()
npu_quant_io_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
npu_quant_io_session = ort.InferenceSession(
    QUANT_IO_MODEL_PATH,
    sess_options=npu_quant_io_options,
    providers=["QNNExecutionProvider"],
    provider_options=[{
        "backend_path": "QnnHtp.dll",
        "htp_performance_mode": "sustained_high_performance",
        "enable_htp_fp16_precision": "1",
        # "profiling_level": "detailed",
        # "profiling_file_path": "npu_quant_io_profile.csv",
    }]
)

# Run the quantized I/O model on the NPU to measure the latency.
start_npu_quant_io = time.time()
for i in range(ITERATIONS):
    npu_quant_io_outputs = npu_quant_io_session.run(
        None, {
            "input0_quant_tensor": input0_quant_tensor,
            "input1_quant_tensor": input1_quant_tensor,
        })
end_npu_quant_io = time.time()
npu_quant_io_output = npu_quant_io_outputs[0]

# Convert the result back into a float tensor.
npu_quant_io_output_float = dequantize_tensor(npu_quant_io_output, matmul_scale, matmul_zero_point)

print("************ Benchmark Results ************")

# Verify that the results are approximately what we'd expect.
print(f"NPU quantized compute, float I/O accuracy difference is {array_msd(cpu_float_output, npu_quant_output):0.4f}")
print(f"NPU quantized compute and I/O accuracy difference is {array_msd(cpu_float_output, npu_quant_io_output_float):0.4f}")

cpu_s = (end_cpu - start_cpu) / ITERATIONS
npu_quant_s = (end_npu_quant - start_npu_quant) / ITERATIONS
npu_quant_io_s = (end_npu_quant_io - start_npu_quant_io) / ITERATIONS

cpu_ms = cpu_s * 1000.0
npu_quant_ms = npu_quant_s * 1000.0
npu_quant_io_ms = npu_quant_io_s * 1000.0

# Derive the ops per second from the latency and number of ops in the model.
cpu_ops_per_second = round(OPS_PER_INFERENCE / cpu_s)
npu_quant_ops_per_second = round(OPS_PER_INFERENCE / npu_quant_s)
npu_quant_io_ops_per_second = round(OPS_PER_INFERENCE / npu_quant_io_s)

rn = ReadableNumber(precision=0, digit_group_size=3)

print(f"CPU took {cpu_ms:0.2f}ms, {rn.of(cpu_ops_per_second)} ops per second")
print(f"NPU (quantized compute, float I/O) took {npu_quant_ms:0.2f}ms, {rn.of(npu_quant_ops_per_second)} ops per second")
print(f"NPU (quantized compute and I/O) took {npu_quant_io_ms:0.2f}ms, {rn.of(npu_quant_io_ops_per_second)} ops per second")
