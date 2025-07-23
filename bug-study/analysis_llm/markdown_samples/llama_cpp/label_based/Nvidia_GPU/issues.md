# Nvidia_GPU - issues

**Total Issues**: 7
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 6

### Label Distribution

- Nvidia GPU: 7 issues
- bug: 3 issues
- bug-unconfirmed: 2 issues
- Vulkan: 1 issues
- stale: 1 issues
- low severity: 1 issues
- enhancement: 1 issues
- help wanted: 1 issues
- roadmap: 1 issues
- generation quality: 1 issues

---

## Issue #N/A: vulkan: rounding differences on Turing

**Link**: https://github.com/ggml-org/llama.cpp/issues/10764
**State**: closed
**Created**: 2024-12-10T15:18:51+00:00
**Closed**: 2024-12-10T20:23:19+00:00
**Comments**: 1
**Labels**: Nvidia GPU, bug-unconfirmed, Vulkan

### Description

### Name and Version

fails at commit 26a8406ba9198eb6fdd8329fa717555b4f77f05f

Not a recent regression, to my knowledge

### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Problem description & steps to reproduce

There are failures in im2col and rope tests that look like rounding differences. I believe Turing is using round-to-zero, which is allowed by the Vulkan spec but doesn't match other implementations or the CPU reference.

```
IM2COL(type_input=f32,type_kernel=f16,dst_type=f16,ne_input=[10,10,3,1],ne_kernel=[3,3,3,1],s0=1,s1=1,p0=1,p1=1,d0=1,d1=1,is_2D=1): [IM2COL] NMSE = 0.000000203 > 0.000000100 �[1;31mFAIL�[0m

ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): [ROPE] NMSE = 0.000000240 > 0.000000100 �[1;31mFAIL�[0m
```

(more failures at https://github.com/ggml-org/ci/tree/results/llama.cpp/26/a8406ba9198eb6fdd8329fa717555b4f77f05f/ggml-6-x86-vul

[... truncated for brevity ...]

---

## Issue #N/A: llama.cpp is slow on GPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/9881
**State**: closed
**Created**: 2024-10-14T11:19:53+00:00
**Closed**: 2024-12-01T01:08:05+00:00
**Comments**: 9
**Labels**: Nvidia GPU, bug-unconfirmed, stale, low severity

### Description

### What happened?

llama.cpp is running slow on NVIDIA A100 80GB GPU

Steps to reproduce:
1. git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
2. mkdir build && cd build
3. cmake .. -DGGML_CUDA=ON
4. make GGML_CUDA=1
3. command:  ./build/bin/llama-cli -m ../gguf_files/llama-3-8B.gguf -t 6912 --ctx-size 50 --n_predict 50 --prompt "There are two persons named ram and krishna"
 Here threads are set to 6912 since GPU has 6912 CUDA cores.

It is slow on gpu compared to cpu.
On gpu  eval time is around 0.07 tokens per second.
Is this expected behaviour or any tweak should be done while building llama.cpp?

### Name and Version

version: 3902 (c81f3bbb)
built with cc (GCC) 11.4.1 20231218 (Red Hat 11.4.1-3) for aarch64-redhat-linux

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
```


---

## Issue #N/A: CUDA graphs break quantized K cache

**Link**: https://github.com/ggml-org/llama.cpp/issues/7492
**State**: closed
**Created**: 2024-05-23T12:11:15+00:00
**Closed**: 2024-05-27T17:33:43+00:00
**Comments**: 5
**Labels**: bug, Nvidia GPU

### Description

As of right now it is already possible on master to quantize the K cache via e.g. `-ctk q8_0`. However, this is currently broken on master for batch size 1. Disabling CUDA graphs via the environment variable `GGML_CUDA_DISABLE_GRAPHS=1` fixes the issue.

cc: @agray3 

---

## Issue #N/A: Significantly different results (and WRONG) inference when GPU is enabled.

**Link**: https://github.com/ggml-org/llama.cpp/issues/7048
**State**: closed
**Created**: 2024-05-02T18:51:50+00:00
**Closed**: 2024-05-17T18:49:39+00:00
**Comments**: 40
**Labels**: bug, Nvidia GPU

### Description

I am running llama_cpp version 0.2.68 on Ubuntu 22.04LTS under conda environment. Attached are two Jupyter notebooks with ONLY one line changed (use CPU vs GPU).  As you can see for exact same environmental conditions switching between CPU/GPU gives vastly different answers where the GPU is completely wrong.  Some pointers on how to debug this I would appreciate it.

The only significant difference between the two files is this one liner
      `#n_gpu_layers=-1, # Uncomment to use GPU acceleration`

The model used was **openhermes-2.5-mistral-7b.Q5_K_M.gguf**

[mistral_llama_large-gpu.pdf](https://github.com/ggerganov/llama.cpp/files/15192723/mistral_llama_large-gpu.pdf)
[mistral_llama_large-cpu.pdf](https://github.com/ggerganov/llama.cpp/files/15192725/mistral_llama_large-cpu.pdf)



---

## Issue #N/A: ggml : add GPU support for Mamba models

**Link**: https://github.com/ggml-org/llama.cpp/issues/6758
**State**: open
**Created**: 2024-04-19T06:47:35+00:00
**Comments**: 32
**Labels**: enhancement, help wanted, Nvidia GPU, roadmap

### Description

Recently, initial Mamba support (CPU-only) has been introduced in #5328 by @compilade 

In order to support running these models efficiently on the GPU, we seem to be lacking kernel implementations for the following 2 ops:

- `GGML_OP_SSM_CONV`
- `GGML_OP_SSM_SCAN`

Creating this issue to keep track of this and give more visibility of this feature. Help with implementing the missing kernels for CUDA and Metal (and other backends potentially) is welcome. We can also discuss if anything else is required to better support this architecture in `llama.cpp`

---

## Issue #N/A: Broken generation with specific ngl values

**Link**: https://github.com/ggml-org/llama.cpp/issues/3820
**State**: closed
**Created**: 2023-10-27T22:49:53+00:00
**Closed**: 2023-11-09T14:08:31+00:00
**Comments**: 9
**Labels**: bug, generation quality, Nvidia GPU

### Description

While playing with implementing compression for copy/save state, I found a bug, which turned out to be reproducible in current `main` (41aee4d)

It seems to be model independent, and no parameters other than `-ngl` seem to make a difference either.

The first symptom happens for `save-load-state`, `main` and `server`, when `-ngl` equal to exactly N-1 is specified, basically this happens (generated output):

```
 Hello there!###############################
```

Second symptom was found by accident, when fiddling with `save-load-state` for the purpose of implementing compression. Basically, if `-ngl` is N or bigger (all layers loaded),
The problem above, seems to disappear, however:
Not only `save-load-state` fails because generated text is different for both runs,
but also, **after** some tokens were sampled `llama_copy_state_data` outputs mostly empty array, which I only noticed because I tried to dump the state post generation, and suddenly started to get 99% compression 

[... truncated for brevity ...]

---

## Issue #N/A: llama : improve batched decoding performance

**Link**: https://github.com/ggml-org/llama.cpp/issues/3479
**State**: closed
**Created**: 2023-10-04T20:20:55+00:00
**Closed**: 2023-10-24T13:48:38+00:00
**Comments**: 12
**Labels**: performance, Nvidia GPU

### Description

Based on info from the following post, [vLLM](https://github.com/vllm-project/vllm) can achieve the following speeds for parallel decoding on A100 GPU:

https://docs.mystic.ai/docs/mistral-ai-7b-vllm-fast-inference-guide

Batch size | Tokens/s
-- | --
1 | 46
10 | 400
60 | 1.8k

(thanks to @wsxiaoys for bringing my attention to this)

Even though `llama.cpp`'s single batch inference is faster ([~72 t/s](https://github.com/ggerganov/llama.cpp/discussions/3359)) we currently don't seem to scale well with batch size. At batch size 60 for example, the performance is roughly x5 slower than what is reported in the post above.

We should understand where is the bottleneck and try to optimize the performance.

```bash
# batch size 1
./parallel -m ~/f16.gguf -t 1 -ngl 100 -c 8192 -b 512 -s 1 -np 1 -ns 128 -n 100 -cb

# batch size 10
./parallel -m ~/f16.gguf -t 1 -ngl 100 -c 8192 -b 512 -s 1 -np 10 -ns 128 -n 100 -cb

# batch size 60
./parallel -m ~/f16.gguf -t 1 -ngl 100 

[... truncated for brevity ...]

---

