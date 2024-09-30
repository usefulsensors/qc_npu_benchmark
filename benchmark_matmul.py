import numpy as np
import onnx
from onnx import helper as h, TensorProto as tp
import onnxruntime as ort
from onnxruntime.quantization import QuantFormat, QuantType, quantize, CalibrationDataReader, quantize_static
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
import time


INPUT0_SHAPE = [1, 6, 1500, 256]
INPUT1_SHAPE = [1, 6, 256, 1500]
OUTPUT_SHAPE = [1, 6, 1500, 1500]
INPUT_RANGE = 1.0 / 5.0

FLOAT_MODEL_PATH = "matmul_model_float.onnx"
QUANT_MODEL_PATH = "matmul_model_quant.onnx"

ITERATIONS = 10

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

def quantize_tensor(input, scale, zero_point):
    assert input.dtype == np.float32
    return np.clip(np.round(input / scale) + zero_point, 0, 255).astype(np.uint8)

def dequantize_tensor(input, scale, zero_point):
    assert input.dtype == np.uint8
    return (input.astype(np.float32) - zero_point) * scale

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

def array_msd(x, y):
    difference = x - y
    msd = np.mean(np.sqrt(difference * difference))
    return msd

matmul_float_model = make_matmul_float_model()
onnx.checker.check_model(matmul_float_model)
onnx.save(matmul_float_model, FLOAT_MODEL_PATH)

# Arbitrary but fixed seed value.
rng = np.random.default_rng(7528840384)
input0_tensor = rng.random((INPUT0_SHAPE)).astype(np.float32) * INPUT_RANGE
input1_tensor = rng.random((INPUT1_SHAPE)).astype(np.float32) * INPUT_RANGE

matmul_float_model_serialized = matmul_float_model.SerializeToString()

cpu_options = ort.SessionOptions()
cpu_session = ort.InferenceSession(
    matmul_float_model_serialized,
    sess_options=cpu_options,
)

start_cpu = time.time()
for i in range(ITERATIONS):
    cpu_float_outputs = cpu_session.run(
        None, {
            "input0_tensor": input0_tensor,
            "input1_tensor": input1_tensor,
        })
end_cpu = time.time()
cpu_float_output = cpu_float_outputs[0]


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

npu_options = ort.SessionOptions()
npu_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
npu_session = ort.InferenceSession(
    QUANT_MODEL_PATH,
    sess_options=npu_options,
    providers=["QNNExecutionProvider"],
    provider_options=[{
        "backend_path": "QnnHtp.dll",
        "htp_performance_mode": "sustained_high_performance",
        "enable_htp_fp16_precision": "1",
        # "profiling_level": "detailed",
        # "profiling_file_path": "matmul_profile.csv",
    }]
)

start_npu = time.time()
for i in range(ITERATIONS):
    npu_float_outputs = npu_session.run(
        None, {
            "input0_tensor": input0_tensor,
            "input1_tensor": input1_tensor,
        })
end_npu = time.time()
npu_float_output = npu_float_outputs[0]

print(f"Accuracy difference is {array_msd(cpu_float_output, npu_float_output):0.4f}")

cpu_ms = 1000.0 * (end_cpu - start_cpu) / ITERATIONS
npu_ms = 1000.0 * (end_npu - start_npu) / ITERATIONS

print(f"CPU took {cpu_ms:0.2f}ms")
print(f"NPU took {npu_ms:0.2f}ms")