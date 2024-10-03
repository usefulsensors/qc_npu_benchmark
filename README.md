# Benchmarking Qualcomm's NPU on the Microsoft Surface Tablet

TL;DR - We see 1.3% of Qualcomm's NPU 45 Teraops/s claim when benchmarking Windows AI PCs

  - [Introduction](#introduction)
  - [Installation](#installation)
    - [Python](#python)
    - [Cmake](#cmake)
    - [Visual Studio](#visual-studio)
    - [Pip Packages](#pip-packages)
  - [Benchmark](#benchmark)
    - [Running](#running)
    - [Understanding the Output](#understanding-the-output)
    - [What the Benchmark Measures](#what-the-benchmark-measures)
    - [Design Decisions](#design-decisions)
      - [Compute Bound](#compute-bound)
      - [Power Settings](#power-settings)
      - [Model Topology](#model-topology)
      - [Configuration Errors](#configuration-errors)
      - [Onnx Framework](#onnx-framework)
  - [Interpreting the Results](#interpreting-the-results)

## Introduction

Microsoft now offers Surface tablets that run Windows on a Qualcomm Arm-based 
SoC. These are marketed as AI PCs, due to their ability to run machine learning
models faster and more efficiently than other systems. We are fans of 
Qualcomm's hardware, and its NPU in particular, so we've invested a lot of time
and resources into porting our third-party app to this plaform.

Unfortunately there  aren't many code examples or benchmarks available to 
demonstrate how to achieve fast results as an external developer, so we've put
together a small standalone project to show the performance we're seeing. It's
significantly below what we'd hoped for, so we're publishing this benchmark to
see if we can get ideas on how to achieve lower latency. I'm hopeful there will
be software changes, either at the application, framework, or driver level, 
that will improve these results in the future, since I've seen the underlying 
hardware perform very effectively on other platforms like Android.

## Installation

### Python

We're using Python to run our test scripts, and on Windows [there are several ways to install the language](https://docs.python.org/3/using/windows.html).
As of October 2nd, 2024, the Python available on the Microsoft Store doesn't
support the Arm architecture, and so it's not suitable for running the packages
we need to access Qualcomm's NPU. Instead, you should use [the official Python dot org installer](https://www.python.org/downloads/). 
For the results reported here I used [version 3.11.9](https://www.python.org/ftp/python/3.11.9/python-3.11.9-arm64.exe).

### Cmake

We'll also need the cmake build tool to compile Onnx (since prebuilt packages
aren't yet available for Windows on Arm). To do this I ran the following
command from a Powershell:

```
winget install cmake
```

### Visual Studio

The build process also requires Visual Studio for the compiler. Download Visual
Studio Community Edition (not Code!) from [visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/).

During the installation you will be prompted to select `Workload` from several options: select `Desktop C++ Development` checkbox then press install.

### Pip Packages

You can install all the required Python packages by running the following
from within this folder:

```
py -m pip install -r requirements.txt
```

This includes a couple of custom packages. The first is [my branch of Onnx](https://github.com/petewarden/onnx/tree/rel-1.16.2),
which has [a fix for compiling using the official `py` launcher](https://github.com/onnx/onnx/pull/6407)
backported to Onnx version 1.16, since the Qualcomm Onnx Runtime doesn't work
with newer Onnx versions (giving an `Unsupported model IR version` error).

I also grab [a nightly build](https://aiinfra.pkgs.visualstudio.com/2692857e-05ef-43b4-ba9c-ccf1c22c437c/_packaging/7982ae20-ed19-4a35-a362-a96ac99897b7/pypi/download/ort-nightly-qnn/1.20.dev20240928001/ort_nightly_qnn-1.20.0.dev20240928001-cp311-cp311-win_arm64.whl#sha256=3b12e3882d1afadf66c2349b2a167dfcbb9ae7a332dc98e0fd51c101d34ddf6e)
of [Qualcomm's Onnx Runtime package](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html). 
If you want to install a more recent version, there's [a list here](https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ort-nightly-qnn/).

## Benchmark

### Running

To execute the benchmark, run:

```
py benchmark_matmul.py
```

### Understanding the Output

The Onnx runtime initially generates a lot of log spam, including:

```
Error in cpuinfo: Unknown chip model name 'Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz'.
Please add new Windows on Arm SoC/chip support to arm/windows/init.c!
unknown Qualcomm CPU part 0x1 ignored
```

and

```
Starting stage: Finalizing Graph Sequence
Completed stage: Finalizing Graph Sequence (115919 us)
Starting stage: Completion
Completed stage: Completion (1025 us)
```

After all those messages, you should see the actual benchmark 
results at the end, something like this:

```bash
************ Benchmark Results ************
NPU quantized compute, float I/O accuracy difference is 0.0100
NPU quantized compute and I/O accuracy difference is 0.0060
CPU took 8.42ms, 821,141,860,688 ops per second
NPU (quantized compute, float I/O) took 30.63ms, 225,667,671,183 ops per second
NPU (quantized compute and I/O) took 12.05ms, 573,475,650,364 ops per second
```

The first two lines confirm that the numerical results of the operations match
between the CPU and the NPU. The final three show the latency of the three
approaches to running a simple model. The latency is the wall time it took to
execute the model from start to finish, and the ops per second is calculated
from that latency to indicate the equivalent computational throughput.

In this example, we see the CPU is capable of running 821 billion ops/second
(821 Gigaops), the first NPU approach gives us 225 Gigaops, and the second 573
Gigaops.

### What the Benchmark Measures

This benchmark is designed to resemble some real world models we depend on,
running 6 large matrix multiplications that are similar to the most 
time-consuming layers in transformer models like OpenAI's Whisper. The shapes
are (6, 1500, 256) X (6, 256, 1500), producing a (6, 1500, 1500) result. The
model we running consists of a single MatMul node with two inputs and one 
output.

The models are created on the fly using the Onnx model framework, and then fed
into the Onnx runtime. The control model is a pure float version that runs
entirely on the CPU.

The NPU mostly requires quantized models to run effectively (though it has
limited support for float16). The first approach we took to quantization used
[the official ORT `quantize_static()` method](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#static-quantization).
For convenience this leaves the input and output tensors in 32-bit float and
performs runtime conversions at the start and end of the graph so that the rest
of the computation happens in eight-bit.

Unfortunately we discovered that the conversion operations as implemented on
the NPU were extremely slow, much slower than the main matrix multiplication
in fact. You can see the results in the `npu_quant_profile.csv` file in this
repository, with conversions taking over 75% of the time.

To work around this, we constructed an equivalent model graph programmatically
with eight-bit inputs and outputs This is the second "quantized compute and 
I/O" approach mentioned in the results. This is usually around three times
faster than the float I/O version, and profiling shows most of the time is
going on the matrix multiplication, as we'd hope.

### Design Decisions

There are a lot of variables involved in measuring performance. Here are some
of the assumptions we've made:

#### Compute Bound 

Modern transformer models are based around large matrix multiplications, unlike
older convolutional models. One potential issue is that accelerators could
become memory bound if the layers start to resemble matrix times vectors, since
that doesn't allow reuse of many of the weights, and performance becomes bottle
necked on fetching values from DRAM. We've tried to avoid that by making both
the input matrices more square, so that tiling and reuse should be possible.

The original matrices from the tiny Whisper model had a k dimension of only 64,
so in case that was too small we bumped it up to 256 in this benchmark to give
as much room for SIMD optimizations as possible.

#### Power Settings

Windows has a lot of different configuration options around energy usage, so we
tried to ensure that all of the settings were on "Best Performance" and that we
ran the benchmark with the tablet connected to mains power. There's also a 
session option on the Qualcomm Onnx Runtime, `htp_performance_mode`, that we 
set to `sustained_high_performance`, since that seemed to give the lowest 
overall latency in our experiments.

#### Model Topology

We wanted to create a graph of operations that reflected modern AI models, but
was simple enough to easily interpret. We could have added multiple layers, or
used convolutions, or static weights, but settled for a single matrix 
multiplication operation with dynamic inputs, since that reflected the 
transformer architectures that are widely used for LLMs and other modern 
models.

#### Configuration Errors

It's possible that the way we build and run our models causes them to fall off
the fast path of the drivers or accelerator implementation. For example, we're
using unsigned eight-bit quantization, with qdq elements in the graph. We've
attempted to follow best practice from the documentation, but we'd welcome ways
to improve performance, especially since these would improve the performance of
our actual applications.

#### Onnx Framework

There are multiple different ways to access AI acceleration on Windows. We 
looked at DirectML, but it only seems to support GPU access. OpenVino doesn't
run on our Arm hardware, as far as we can tell. We've seen similar performance
results to those shown here using the [Qualcomm QNN SDK](https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai) 
directly. TensorFlow Lite isn't supported on Windows for Arm. From this 
research and our experiments, Onnx is supported by both Microsoft and Qualcomm,
and seems to be the best framework to use to get accelerated performance from
the NPU, but we're interested in learning if other APIs would be more 
appropriate.

## Interpreting the Results

The results shown here are current as of October 2nd, 2024, when running on a
Microsoft Surface Pro 11th Edition, with a Snapdragon(R) X 12-core X1E80100
clocked at 3.40 GHz. The first obvious thing is that the NPU results, even
without float conversion, are slower than the CPU. This is not ideal for an
accelerator, even though it could still potentially offer energy or sustained
performance advantages that make it worth using.

The second conclusion is that the measured performance of 573 billion 
operations per second is only 1.3% of the 45 trillion ops/s that [the marketing material](https://www.microsoft.com/en-us/surface/devices/surface-pro-11th-edition)
promises.
