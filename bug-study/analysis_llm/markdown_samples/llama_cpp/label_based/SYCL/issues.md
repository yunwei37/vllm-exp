# SYCL - issues

**Total Issues**: 3
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 3

### Label Distribution

- bug-unconfirmed: 3 issues
- stale: 3 issues
- SYCL: 3 issues
- critical severity: 1 issues
- medium severity: 1 issues
- high severity: 1 issues

---

## Issue #N/A: Bug: Intel Arc - not working at all

**Link**: https://github.com/ggml-org/llama.cpp/issues/9106
**State**: closed
**Created**: 2024-08-20T19:45:26+00:00
**Closed**: 2024-12-17T01:07:43+00:00
**Comments**: 28
**Labels**: bug-unconfirmed, stale, SYCL, critical severity

### Description

### What happened?

Going through the manual - SYCL I mean. Everything compiles okay. Running it always thows an error. Can't make it work. OS used: Linux Gentoo. P.S. docker doesn't work either. P.P.S. device IS listed in the list.

### Name and Version

# ./build/bin/llama-cli --version
version: 3609 (2f3c1466)
built with Intel(R) oneAPI DPC++/C++ Compiler 2024.2.1 (2024.2.1.20240711) for x86_64-unknown-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
# ZES_ENABLE_SYSMAN=1 ./build/bin/llama-cli -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33 -sm none -mg 0
Log start
main: build = 3609 (2f3c1466)
main: built with Intel(R) oneAPI DPC++/C++ Compiler 2024.2.1 (2024.2.1.20240711) for x86_64-unknown-linux-gnu
main: seed  = 1724182694
llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from models/llama-2-7b.Q4_0.gguf (version GGUF V2)
llama_

[... truncated for brevity ...]

---

## Issue #N/A: Bug: InvalidModule: Invalid SPIR-V module: input SPIR-V module uses extension 'SPV_INTEL_memory_access_aliasing' which were disabled by --spirv-ext option

**Link**: https://github.com/ggml-org/llama.cpp/issues/8551
**State**: closed
**Created**: 2024-07-18T04:49:20+00:00
**Closed**: 2024-09-06T01:07:04+00:00
**Comments**: 17
**Labels**: bug-unconfirmed, stale, SYCL, medium severity

### Description

### What happened?

Currently on Fedora 40 with Intel Arc A750.

Running the following:
```txt
ZES_ENABLE_SYSMAN=1 ./build/bin/llama-server \
-t 10 \
-ngl 20 \
-b 512 \
--ctx-size 16384 \
-m ~/llama-models/llama-2-7b.Q4_0.gguf \
--color -c 3400 \
--seed 42 \
--temp 0.8 \
--top_k 5 \
--repeat_penalty 1.1 \
--host :: \
--port 8080 \ 
-n -1 \
-sm none -mg 0
```

gives the following output:
```txt
INFO [                    main] build info | tid="140466031364096" timestamp=1721277895 build=3411 commit="e02b597b"
INFO [                    main] system info | tid="140466031364096" timestamp=1721277895 n_threads=10 n_threads_batch=-1 total_threads=28 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 0 | "
llama_model_loader: loaded meta data with

[... truncated for brevity ...]

---

## Issue #N/A: Bug: [SYCL] Inference not working correctly on multiple GPUs

**Link**: https://github.com/ggml-org/llama.cpp/issues/8294
**State**: closed
**Created**: 2024-07-04T11:46:57+00:00
**Closed**: 2024-09-07T01:07:06+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, stale, SYCL, high severity

### Description

### What happened?

I am using Llama.cpp + SYCL to perform inference on a multiple GPU server. However, I get a Segmentation Fault when using multiple GPUs. The same model can produce inference output correctly with single GPU mode.

```shell
git clone https://github.com/ggerganov/llama.cpp.git
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release -j -v

cd ~/llama.cpp/
./build/bin/llama-ls-sycl-device

## single gpu, ok
./build/bin/llama-cli -m ~/mistral-7b-v0.1.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 33 -s 0 -sm none -mg 0

## multiple gpus, Segmentation Fault, core dumped
./build/bin/llama-cli -m ~/mistral-7b-v0.1.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 33 -s 0 -sm layer
``` 

![image](https://github.com/ggerganov/llama.cpp/assets/16000946/f2ef0a11-6b54-43b2-acba-23624e1

[... truncated for brevity ...]

---

