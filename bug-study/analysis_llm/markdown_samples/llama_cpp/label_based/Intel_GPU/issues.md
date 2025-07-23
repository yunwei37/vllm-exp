# Intel_GPU - issues

**Total Issues**: 9
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 9

### Label Distribution

- Intel GPU: 9 issues
- bug-unconfirmed: 6 issues
- stale: 4 issues

---

## Issue #N/A: ignore : test sub-issue

**Link**: https://github.com/ggml-org/llama.cpp/issues/11276
**State**: closed
**Created**: 2025-01-17T09:19:10+00:00
**Closed**: 2025-01-17T09:19:57+00:00
**Comments**: 0
**Labels**: Intel GPU

### Description

Just testing "Github sub-issues" - please ignore

---

## Issue #N/A: Llama.cpp not working with intel ARC 770?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7042
**State**: closed
**Created**: 2024-05-02T12:37:39+00:00
**Closed**: 2024-05-08T19:23:42+00:00
**Comments**: 17
**Labels**: Intel GPU

### Description

Hi,

I am trying to get llama.cpp to work on a workstation with one ARC 770 Intel GPU but somehow whenever I try to use the GPU, llama.cpp does something (I see the GPU being used for computation using intel_gpu_top) for 30 seconds or so and then just hang there, using 100% CPU (but only one core) as if it would be waiting for something to happen...


I am using the following command:

```sh
ZES_ENABLE_SYSMAN=0 ./main -m ~/LLModels/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q5_K_S.gguf -p "Alice lived in Wonderland and her favorite food was:" -n 512 -e -ngl 33 -sm none -mg 0 -t 32
```

it doesn't matter if I ignore the ZES_ENABLE_SYSMAN part.

now, the same biuld does work when `-ngl 0`. There I see the 32 cores be used and the model produces output.

If I run `clinfo`, I get the following output.

```sh
Platform #0: Intel(R) FPGA Emulation Platform for OpenCL(TM)
 `-- Device #0: Intel(R) FPGA Emulation Device
Platform #1: Intel(R) OpenCL
 `-- Device #0

[... truncated for brevity ...]

---

## Issue #N/A: SYCL build failed

**Link**: https://github.com/ggml-org/llama.cpp/issues/5547
**State**: closed
**Created**: 2024-02-17T11:48:36+00:00
**Closed**: 2024-05-30T01:23:41+00:00
**Comments**: 11
**Labels**: bug-unconfirmed, Intel GPU, stale

### Description

The build can be completed, but after it is finished
Run build\bin\main.exe
```
main: build = 0 (unknown)
main: built with MSVC 19.39.33519.0 for
main: seed  = 1708170078
llama_model_load: error loading model: failed to open models/7B/ggml-model-f16.gguf: No such file or directory
llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model 'models/7B/ggml-model-f16.gguf'
main: error: unable to load model
```
Run .\examples\sycl\win-run-llama2.bat
```
:: oneAPI environment initialized ::
warning: not compiled with GPU offload support, --n-gpu-layers option will be ignored
warning: see main README.md for information on enabling GPU BLAS support
```
My PC：
OS:Windows 11 (22631.3155)
CPU:AMD Ryzen 5 5600X
GPU:Intel Arc A770

Run sycl-ls
```
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.12.0.12_195853.xmain-hotfix]
[opencl:cpu:1] Intel(R) OpenCL, AMD Ry

[... truncated for brevity ...]

---

## Issue #N/A: [SYCL] Segmentation fault after #5411

**Link**: https://github.com/ggml-org/llama.cpp/issues/5469
**State**: closed
**Created**: 2024-02-13T02:11:58+00:00
**Closed**: 2024-02-21T09:52:08+00:00
**Comments**: 30
**Labels**: bug-unconfirmed, Intel GPU

### Description

System: Arch Linux,
CPU: Intel i3 12th gen
GPU: Intel Arc A750
RAM: 16GB

llama.cpp version: b2134


Previously the build was failing with `-DLLAMA_SYCL_F16=ON` which has been fixed in #5411. Upon running this build, it crashes with segmentation fault.

logs:
```
bin/main -m ~/Public/Models/Weights/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf  -p "hello " -n 1000 -ngl 99
Log start
main: build = 2134 (099afc62)
main: built with Intel(R) oneAPI DPC++/C++ Compiler 2024.0.0 (2024.0.0.20231017) for x86_64-unknown-linux-gnu
main: seed  = 1707789832
GGML_SYCL_DEBUG=0
ggml_init_sycl: GGML_SYCL_F16:   yes
ggml_init_sycl: SYCL_USE_XMX: yes
found 4 SYCL devices:
  Device 0: Intel(R) Arc(TM) A750 Graphics,	compute capability 1.3,
	max compute_units 448,	max work group size 1024,	max sub group size 32,	global mem size 8096681984
  Device 1: Intel(R) FPGA Emulation Device,	compute capability 1.2,
	max compute_units 4,	max work group size 67108864,	max sub group size 64,	global mem si

[... truncated for brevity ...]

---

## Issue #N/A: Unable to build llama.cpp on Intel DevCloud

**Link**: https://github.com/ggml-org/llama.cpp/issues/5439
**State**: closed
**Created**: 2024-02-10T12:49:32+00:00
**Closed**: 2024-04-20T01:07:06+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, Intel GPU, stale

### Description

Executed the following commands, unable to get build sucessfully:

`mkdir -p build`
`cd build\n`
`cmake .. -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx`

Getting this error 
![image](https://github.com/ggerganov/llama.cpp/assets/95060707/9041cef8-3c4e-433f-901c-1aa147bb5a2a)


---

## Issue #N/A: Running llama.cpp with sycl in Docker fails with "Unknown PII Error"

**Link**: https://github.com/ggml-org/llama.cpp/issues/5400
**State**: closed
**Created**: 2024-02-07T21:29:17+00:00
**Closed**: 2024-02-08T11:02:46+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, Intel GPU

### Description

I have an Intel Arc 770, I'm trying to run llama.cpp with Docker following https://github.com/ggerganov/llama.cpp/blob/master/README-sycl.md but it fails with:

```
Native API failed. Native API returns: -999 (Unknown PI error) -999 (Unknown PI error)
Exception caught at file:/app/ggml-sycl.cpp, line:14735, func:operator()
SYCL error: CHECK_TRY_ERROR((*stream) .memcpy((char *)tensor->data + offset, data, size) .wait()): Meet error in this line code!
  in function ggml_backend_sycl_buffer_set_tensor at /app/ggml-sycl.cpp:14735
GGML_ASSERT: /app/ggml-sycl.cpp:2919: !"SYCL error"
```

I've been trying also to run it with:

```
docker run -it --rm -v "$(pwd):/app:Z" --device /dev/dri llama-cpp-sycl -m "/app/models/c0c3c83d0ec33ffe925657a56b06771b" -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33  

```

I'm also trying that with LocalAI where I have been creating manually a container with sycl, however there the error is different (PR https://githu

[... truncated for brevity ...]

---

## Issue #N/A: SYCL backend support Multi-card

**Link**: https://github.com/ggml-org/llama.cpp/issues/5282
**State**: closed
**Created**: 2024-02-02T12:01:33+00:00
**Closed**: 2024-03-05T15:43:14+00:00
**Comments**: 3
**Labels**: Intel GPU

### Description

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/5277

<div type='discussions-op-text'>

<sup>Originally posted by **airMeng** February  2, 2024</sup>
Feel free to drop a note, let's know if you have any feature request or bugs (even unconfirmed)

- [ ] Multi-card Support
- [ ] Multi-batch Support [#5272](https://github.com/ggerganov/llama.cpp/issues/5272)
- [ ] CI test error for more than one GPU is detected and used.
  Current code returns all SYCL devices, including CPU, GPU (level-zero, opencl), FPGA. SYCL only support GPU. So when CI test on other devices, it will be fault.
- [ ] Support no-mmap parameter in other application. 
  There is known issue of SYCL: memcpy() from host (mmap) to device will hang in same cases. It's not resolved now. A work around solution is no use mmap. I have handled it in llama-bench (add --mmap parameter). We need add to more applications in examples.
- [ ] Clean code for warning and unused macro and variable.
  Sugges

[... truncated for brevity ...]

---

## Issue #N/A: Excessively slow prompt processing time with 70B partially offloaded in SYCL

**Link**: https://github.com/ggml-org/llama.cpp/issues/5272
**State**: closed
**Created**: 2024-02-02T04:38:28+00:00
**Closed**: 2024-05-09T01:06:27+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, Intel GPU, stale

### Description

prompt processing is extremely slow with a 70B partially offloaded.
`llama-bench.exe -ngl 20 -m "D:\models\lzlv_70b_fp16_hf.Q4_K_M.gguf"`
Using device 0 (Intel(R) Arc(TM) A770 Graphics) as main device
| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| llama 70B Q4_K - Medium        |  38.58 GiB |    68.98 B | SYCL       |  20 | pp 512     |      2.14 ± 0.28 |
| llama 70B Q4_K - Medium        |  38.58 GiB |    68.98 B | SYCL       |  20 | tg 128     |      1.03 ± 0.01 |

build: a28c5eff (2045)



---

## Issue #N/A: Build option -DLLAMA_SYCL_F16=ON is ignored on Windows when building with SYCL

**Link**: https://github.com/ggml-org/llama.cpp/issues/5271
**State**: closed
**Created**: 2024-02-02T03:53:07+00:00
**Closed**: 2024-04-20T01:07:09+00:00
**Comments**: 8
**Labels**: bug-unconfirmed, Intel GPU, stale

### Description

Didn't notice this before, but when building with SYCL on Windows, -DLLAMA_SYCL_F16=ON is ignored when built following the instructions given in the README.

---

