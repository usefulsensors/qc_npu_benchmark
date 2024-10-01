# qc_windows_onnx_example
Code sample showing how to run and benchmark models on Qualcomm's Window PCs

Install py 3.11.9
winget install cmake
py -m pip install git+https://github.com/petewarden/onnx@rel-1.16.2
py -m pip install https://aiinfra.pkgs.visualstudio.com/2692857e-05ef-43b4-ba9c-ccf1c22c437c/_packaging/7982ae20-ed19-4a35-a362-a96ac99897b7/pypi/download/ort-nightly-qnn/1.20.dev20240928001/ort_nightly_qnn-1.20.0.dev20240928001-cp311-cp311-win_arm64.whl#sha256=3b12e3882d1afadf66c2349b2a167dfcbb9ae7a332dc98e0fd51c101d34ddf6e

```bash
NPU quantized compute, float I/O accuracy difference is 0.0100
NPU quantized compute and I/O accuracy difference is 0.0060
CPU took 8.42ms, 821,141,860,688 ops per second
NPU (quantized compute, float I/O) took 30.63ms, 225,667,671,183 ops per second
NPU (quantized compute and I/O) took 12.05ms, 573,475,650,364 ops per second
```