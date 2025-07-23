# quick_1to24hours - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- bug-unconfirmed: 17 issues
- high severity: 3 issues
- medium severity: 2 issues
- model: 2 issues
- critical severity: 2 issues
- documentation: 1 issues
- high priority: 1 issues
- low severity: 1 issues
- bug: 1 issues
- hardware: 1 issues

---

## Issue #N/A: Bug: infill reference crashed

**Link**: https://github.com/ggml-org/llama.cpp/issues/8138
**State**: closed
**Created**: 2024-06-26T13:54:35+00:00
**Closed**: 2024-06-27T07:46:42+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

./llama-infill -t 10 -ngl 0 -m ../../models/Publisher/Repository/codellama-13b.Q3_K_S.gguf --temp 0.7 --repeat_penalty 1.1 -n 20 --in-prefix "def helloworld():\n    print(\"hell" --in-suffix "\n   print(\"goodbye world\")\n    "

this command causes llama.cpp abort

### Name and Version

./llama-llava-cli --version
version: 3235 (88540445)
built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.5.0

### What operating system are you seeing the problem on?

Mac

### Relevant log output

```shell
Log start
main: build = 3235 (88540445)
main: built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.5.0
main: seed  = 1719410022
llama_model_loader: loaded meta data with 20 key-value pairs and 363 tensors from ../../models/Publisher/Repository/codellama-13b.Q3_K_S.gguf (version GGUF V2)
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:

[... truncated for brevity ...]

---

## Issue #N/A: How to get the best out of my dual GPU or GPU+CPU?

**Link**: https://github.com/ggml-org/llama.cpp/issues/3064
**State**: closed
**Created**: 2023-09-07T15:09:30+00:00
**Closed**: 2023-09-07T20:08:55+00:00
**Comments**: 4

### Description

Hi, Sorry to ask here, but I can't access discord.

Can anyone advise me as to the best models and options for my system: 

## Arch linux, Amd 5950x 64GB ram, Nvidia 3060 12GB, and 1080 8GB.

I prefer quality over speed, as I plan on running this for special tasks in a queue.

Thanks!

---

## Issue #N/A: Bug: on AMD gpu, it offloads all the work to the CPU unless you specify --n-gpu-layers on the llama-cli command line

**Link**: https://github.com/ggml-org/llama.cpp/issues/8164
**State**: closed
**Created**: 2024-06-27T12:19:01+00:00
**Closed**: 2024-06-28T04:22:31+00:00
**Comments**: 24
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

I spent days trying to figure out why it running a llama 3 instruct model was going super slow (about 3 tokens per second on fp16 and 5.6 on 8 bit) on an AMD MI50 32GB using rocBLAS for ROCm 6.1.2, using 0% GPU and 100% cpu even while using some vram.  I'm currently using release b3246

Finally I noticed that (for 8 bit) it said:
```

llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/33 layers to GPU
llm_load_tensors:        CPU buffer size =  8137.64 MiB

```

adding something to the command line like "--n-gpu-layers 100" changed it to
```
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:      ROCm0 buffer size =  7605.33 MiB
llm_load_tensors:        CPU buffer size =   532.31 MiB
```
and it jumped from
```
llama_print_timings:        load time =    1955.34 ms
llama_print_timings:  

[... truncated for brevity ...]

---

## Issue #N/A: Invalid model error : too old, regenerate your model files!

**Link**: https://github.com/ggml-org/llama.cpp/issues/361
**State**: closed
**Created**: 2023-03-21T16:51:17+00:00
**Closed**: 2023-03-22T05:54:53+00:00
**Comments**: 14
**Labels**: documentation, model

### Description

Downloaded Alpaca 7B model successfully using the following command as mentioned in README.md:
`curl -o ./models/ggml-alpaca-7b-q4.bin -C - https://gateway.estuary.tech/gw/ipfs/QmUp1UGeQFDqJKvtjbSYPBiZZKRjLp8shVP9hT8ZB9Ynv1`

When I try to execute the command:
`main -m ./models/ggml-alpaca-7b-q4.bin --color -f ./prompts/alpaca.txt -ins`

This is the error output:
main: seed = 1679417098
llama_model_load: loading model from './models/ggml-alpaca-7b-q4.bin' - please wait ...
llama_model_load: invalid model file './models/ggml-alpaca-7b-q4.bin' (too old, regenerate your model files!)
main: failed to load model from './models/ggml-alpaca-7b-q4.bin'

How to fix this? Is the downloaded model corrupted and should I download it again? What is the SHA1 hash of the model so that I can verify that the downloaded model is corrupted or not?

---

## Issue #N/A: Eval bug: Output NAN when use Qwen3 embedding models with FP16

**Link**: https://github.com/ggml-org/llama.cpp/issues/13795
**State**: closed
**Created**: 2025-05-26T08:58:30+00:00
**Closed**: 2025-05-26T11:03:55+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

version: 5489 (2d38b6e4)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

CPU, CUDA

### Hardware

INTEL(R) XEON(R) PLATINUM 8558 + Hopper GPU

### Models

Qwen3-Embedding-8B

### Problem description & steps to reproduce

1. `git clone https://github.com/ggml-org/llama.cpp`
2. `cmake -B build && cmake --build build --config Release -j 32`
3. `python convert_hf_to_gguf.py Qwen3-Embedding-8B --outfile Qwen3-Embedding-8B-f16.gguf --outtype f16`
```
INFO:hf-to-gguf:Loading model: Qwen3-Embedding-8B
INFO:hf-to-gguf:Model architecture: Qwen3ForCausalLM
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only
INFO:hf-to-gguf:Exporting model...
INFO:hf-to-gguf:gguf: loading model weight map from 'model.safetensors.index.json'
INFO:hf-to-gguf:gguf: loading model part 'model-00001-of-00007.safetensors'
INFO:hf-to-gguf:token_embd.weight,         torch.float32 --> F16, shape = {4096, 151665}

[... truncated for brevity ...]

---

## Issue #N/A: Bug: llama.cpp binaries are compiled dynamically and the library is missing!

**Link**: https://github.com/ggml-org/llama.cpp/issues/8161
**State**: closed
**Created**: 2024-06-27T11:06:26+00:00
**Closed**: 2024-06-28T10:49:18+00:00
**Comments**: 9
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

$ ./build/bin/llama-quantize -h
./build/bin/llama-quantize: error while loading shared libraries: libllama.so: cannot open shared object file: No such file or directory

### Name and Version

latest

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
see up.
```


---

## Issue #N/A: ci failing on main branch

**Link**: https://github.com/ggml-org/llama.cpp/issues/7403
**State**: closed
**Created**: 2024-05-20T00:58:44+00:00
**Closed**: 2024-05-20T15:11:40+00:00
**Comments**: 1
**Labels**: high priority, bug-unconfirmed

### Description

* https://github.com/ggerganov/llama.cpp/pull/7358
    - Server / server (THREAD, Debug) (push) Failing after 26m
        - https://github.com/ggerganov/llama.cpp/actions/runs/9141008261/job/25134947389 
        -  20 - test-backend-ops (Failed)
* https://github.com/ggerganov/llama.cpp/pull/7374
    - ggml-org / ggml-2-x86-cpu - failure 1 in 3:34.28
        - https://github.com/ggml-org/ci/tree/results/llama.cpp/1e/a2a0036e88172d6c8bf7e1a1989a03894dc955/ggml-2-x86-cpu#test_scripts_debug
        - ?? 
* https://github.com/ggerganov/llama.cpp/pull/7395
    - ggml-org / ggml-4-x86-cuda-v100 - failure 8 in 7:00.84
        - https://github.com/ggml-org/ci/tree/results/llama.cpp/d3/59f30921a9f62a0fd299c412ff3f270286fea6/ggml-4-x86-cuda-v100
        -  20 - test-backend-ops (Failed)

---

## Issue #N/A: sh: 1: ./llama.cpp/llama-quantize: not found

**Link**: https://github.com/ggml-org/llama.cpp/issues/8107
**State**: closed
**Created**: 2024-06-25T09:30:17+00:00
**Closed**: 2024-06-25T11:41:09+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

For converting the FP16.gguf to q5_k_m for llama3 i was getting the error **sh: 1: ./llama.cpp/llama-quantize: not found** previously i used **os.system("./llama.cpp/llama-quantize " + gguf_dir + "/" + gguf_F16_name + " " + model_path + " " + m)** in my script to convert even that is not working now its giving the same not found issue

### Name and Version

Lastest pervsion

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
bash: ./llama.cpp/quantize: No such file or directory
```


---

## Issue #N/A: Performance bug: Android aarch64 Neon Performance Regression and i8mm Detection Issues in New Version of llama.cpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/10662
**State**: closed
**Created**: 2024-12-04T19:11:49+00:00
**Closed**: 2024-12-04T20:20:06+00:00
**Comments**: 11
**Labels**: bug-unconfirmed

### Description

### Name and Version

version: 4248 (3b4f2e33) built with clang version 19.1.4 for aarch64-unknown-linux-android24

### Operating systems

Linux

### GGML backends

CPU

### Hardware

Device - Zenfone 9:  - Qualcomm® Snapdragon® 8+ Gen 1 Mobile Platform
```
system_info: n_threads = 4 (n_threads_batch = 4) / 8 | CPU : NEON = 1 | ARM_FMA = 1 | AARCH64_REPACK = 1 |
```
```
lscpu
Architecture:           aarch64
  CPU op-mode(s):       32-bit, 64-bit
  Byte Order:           Little Endian
CPU(s):                 8
  On-line CPU(s) list:  0-7
Vendor ID:              ARM
  Model name:           Cortex-A510
    Model:              3
    Thread(s) per core: 1
    Core(s) per socket: 4
    Socket(s):          1
    Stepping:           r0p3
    CPU(s) scaling MHz: 77%
    CPU max MHz:        2016.0000
    CPU min MHz:        307.2000
    BogoMIPS:           38.40
    Flags:              fp asimd evtstrm aes pmull sha1
                        sha2 crc32 atomics 

[... truncated for brevity ...]

---

## Issue #N/A: [Solved]Model generation speed significantly slows down when using MiroStat V2

**Link**: https://github.com/ggml-org/llama.cpp/issues/12220
**State**: closed
**Created**: 2025-03-06T07:57:53+00:00
**Closed**: 2025-03-06T15:57:10+00:00
**Comments**: 1

### Description

llama.cpp-b4835 Linux CPU

#### ./llama-cli -m Phi-4-mini-instruct-Q5_K_M.gguf --threads 16 --ctx-size 16000 -mli -co -cnv

```
llama_perf_sampler_print:    sampling time =     147.25 ms /   456 runs   (    0.32 ms per token,  3096.75 tokens per second)
llama_perf_context_print:        load time =    2054.69 ms
llama_perf_context_print: prompt eval time =     667.05 ms /    12 tokens (   55.59 ms per token,    17.99 tokens per second)
llama_perf_context_print:        eval time =   40436.85 ms /   443 runs   (   91.28 ms per token,    10.96 tokens per second)
llama_perf_context_print:       total time =   88806.27 ms /   455 tokens
```


#### ./llama-cli -m Phi-4-mini-instruct-Q5_K_M.gguf --threads 16 --ctx-size 16000 --mirostat 2 -mli -co -cnv

```
llama_perf_sampler_print:    sampling time =   17418.53 ms /   508 runs   (   34.29 ms per token,    29.16 tokens per second)
llama_perf_context_print:        load time =    2014.50 ms
llama_perf_context_print: prompt eval time =    1372.56 

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: bartowski/functionary-small-v3.2-GGUF:Q4_K_M model prepends "assistant\n" to text responses when tools are provided

**Link**: https://github.com/ggml-org/llama.cpp/issues/12213
**State**: closed
**Created**: 2025-03-05T22:35:12+00:00
**Closed**: 2025-03-06T00:43:56+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Name and Version

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070 Laptop GPU, compute capability 8.9, VMM: yes
version: 4783 (a800ae46)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

i9-13900HX + NVIDIA GeForce RTX 4070

### Models

https://huggingface.co/bartowski/functionary-small-v3.2-GGUF/blob/main/functionary-small-v3.2-Q4_K_M.gguf

### Problem description & steps to reproduce

`docker run --gpus all --rm --name llama.cpp -p 8080:8080 -v /etc/ssl/certs:/etc/ssl/certs:ro -v /home/ed/.llama.cpp/models:/root/.cache ghcr.io/ggml-org/llama.cpp:full-cuda -s --ctx-size 0 --jinja -fa -hf bartowski/functionary-small-v3.2-GGUF:Q4_K_M --host 0.0.0.0 -ngl 10 --verbose`

```
curl http://localhost:8080/v1/chat/completions -d '{
"model": "gpt-3.5-turbo",
"messages": [
    {
    "role": "

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Fail to compile after commit 202084d31d4247764fc6d6d40d2e2bda0c89a73a

**Link**: https://github.com/ggml-org/llama.cpp/issues/9554
**State**: closed
**Created**: 2024-09-19T21:37:02+00:00
**Closed**: 2024-09-20T16:35:37+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

Compilation fails on CUDA 11 on any version after (and including) commit 202084d31d4247764fc6d6d40d2e2bda0c89a73a, which I've tracked down via git bisect.
In case it may be useful, this is the output of nvcc --version:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```
Operating system is Pop!_OS jammy 22.04 x86_64

The build command used is:
```
make -j GGML_CUDA=1 GGML_CUDA_MMV_Y=2 GGML_DISABLE_LOGS=1 CUDA_DOCKER_ARCH=sm_86
```
from a clean directory. Compilation fails independently of me setting GGML_CUDA_MMV_Y=2.

### Name and Version

version: 3694 (202084d)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
c++ -std=c++11 -fPIC -O3 -g -Wal

[... truncated for brevity ...]

---

## Issue #N/A: Bug:  error loading model architecture: unknown model architecture: 'clip'

**Link**: https://github.com/ggml-org/llama.cpp/issues/7799
**State**: closed
**Created**: 2024-06-06T12:02:26+00:00
**Closed**: 2024-06-07T03:00:50+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

Running llava-v1.6 results in the following error:
`error loading model: error loading model architecture: unknown model architecture: 'clip'`

The command I ran was:

`llama --log-enable --model models/llava-v1.6-mistral-7b.Q5_K_M.mmproj-f16.gguf --mmproj models/llava-v1.6-mistral-7b.Q5_K_M.mmproj-f16.gguf --image media/llama0-banner.png -p "what is in this image?"`

I had no issues running ShareGPT4V
` llama --log-enable --model models/ShareGPT4V-f16.gguf --mmproj models/ShareGPT4V-f16-mmproj.gguf --image media/llama0-banner.png -p "what is in this image?"`
```
generate: n_ctx = 4096, n_batch = 2048, n_predict = -1, n_keep = 1


 what is in this image?
 What does it show? [end of text]
```



### Name and Version

main: build = 3089 (c90dbe0)
main: built with gcc (GCC) 12.3.0 for x86_64-unknown-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
Log start
main: build = 3089 (c90dbe0)
mai

[... truncated for brevity ...]

---

## Issue #N/A: Bug: rwkv and mamba models cannot be used with `-ngl 0` after CPU backend refactor

**Link**: https://github.com/ggml-org/llama.cpp/issues/10351
**State**: closed
**Created**: 2024-11-17T02:47:58+00:00
**Closed**: 2024-11-17T11:25:46+00:00
**Comments**: 1
**Labels**: bug, medium severity

### Description

### What happened?

```
$ ./build/bin/llama-bench -m ~/Downloads/mamba-2.8b-q4_0.gguf -ngl 0
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
/Users/molly/llama.cpp/ggml/src/ggml-backend.cpp:745: pre-allocated tensor in a backend that cannot run the operation
[1]    13345 abort      ./build/bin/llama-bench -m ~/Downloads/mamba-2.8b-q4_0.gguf -ngl 0
```
```
$ ./build/bin/llama-bench -m /Volumes/grouped/Models/rwkv/v6-Finch-7B-HF/v6-Finch-7B-HF-Q4_0.gguf -ngl 0
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
/Users/molly/llama.cpp/ggml/src/ggml-backend.cpp:745: pre-allocated tensor in a ba

[... truncated for brevity ...]

---

## Issue #N/A: Llama.cpp ./main hangs on line "llama_new_context_with_model: graph splits (measure): 1"

**Link**: https://github.com/ggml-org/llama.cpp/issues/5909
**State**: closed
**Created**: 2024-03-06T18:12:46+00:00
**Closed**: 2024-03-06T21:40:07+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

Llama version b2354

I am trying to inference on a HPC cluster instance with one A100 connected to it. I have compiled the binary with `-DLLAMA_CUBLAS=ON`. 
However, every time i try to run llama.cpp, the program hangs on `llama_new_context_with_model: graph splits (measure): 1`

Here is the full code output:
```Log start
main: build = 2354 (e25fb4b1)
main: built with cc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-10) for x86_64-redhat-linux
main: seed  = 1709748105
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA A100 80GB PCIe, compute capability 8.0, VMM: yes
llama_model_loader: loaded meta data with 23 key-value pairs and 164 tensors from /scratch/wtosbor/gemma.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str             

[... truncated for brevity ...]

---

## Issue #N/A: CPU load balancing is far worse for Mixtral / MoE inference vs dense model

**Link**: https://github.com/ggml-org/llama.cpp/issues/4817
**State**: closed
**Created**: 2024-01-08T01:53:07+00:00
**Closed**: 2024-01-08T05:31:40+00:00
**Comments**: 3
**Labels**: bug-unconfirmed

### Description

If I set 6 threads (both BLAS and regular) for Nous Hermes 34b [4_K_M], which is a dense model, I seem to get optimal prompt evaluation speeds compared to 10 or 4 threads, and the CPU load balancing looks like:
![image](https://github.com/ggerganov/llama.cpp/assets/66376113/1674b54d-4f56-4bc7-8054-fb791730567a)

This is equivalent to the amount of performance cores I have on this processor, so this seems to make sense. I tested 4 and 6 threads and they were both worse. However:

<img width="527" alt="image" src="https://github.com/ggerganov/llama.cpp/assets/66376113/5b8a34ef-b1dd-4b22-b60d-e4fc53b3f040">

The load balancing is significantly less even for the batching in Sparse MoE, so overall utilization suffers [even though this is pure CPU inference] on OpenBLAS. [~60% average utilization]

Turning up the BLAS thread count doesn't help either; paradoxically, the net utilization seems to be _worse_ if you turn up the thread count, just like it was for a dense model.

If I r

[... truncated for brevity ...]

---

## Issue #N/A: FP16 and 4-bit quantized model both produce garbage output on M1 8GB

**Link**: https://github.com/ggml-org/llama.cpp/issues/137
**State**: closed
**Created**: 2023-03-14T17:05:51+00:00
**Closed**: 2023-03-14T20:54:06+00:00
**Comments**: 4
**Labels**: hardware

### Description

Both the `ggml-model-q4_0` and `ggml-model-f16` produce a garbage output on my M1 Air 8GB, using the 7B LLaMA model. I've seen the quantized model having problems but I doubt the quantization is the issue as the non-quantized model produces the same output.

```
➜  llama.cpp git:(master) ./main -m ./models/7B/ggml-model-f16.bin -p "Building a website can be done in 10 simple steps:" -t 8 -n 512
main: seed = 1678812348
llama_model_load: loading model from './models/7B/ggml-model-f16.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 1
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: ggml ctx size = 13365.09 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: loading model part 1/1 from '.

[... truncated for brevity ...]

---

## Issue #N/A: tokenizer is converting spaces to ▁ (U+2581)

**Link**: https://github.com/ggml-org/llama.cpp/issues/2743
**State**: closed
**Created**: 2023-08-23T14:06:36+00:00
**Closed**: 2023-08-24T09:26:02+00:00
**Comments**: 3

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Decoding the token 1678 to three spaces.

# Current Behavior

Token 1678 gets decoded in to " ▁▁" (U+0020 U+2581 U+2581). Other tokens with spaces also have the issue.

# Environment and Context

Run main like this
```
$ ./main -m models/llama-2-13b.q6_K.ggu

[... truncated for brevity ...]

---

## Issue #N/A: can't compile main

**Link**: https://github.com/ggml-org/llama.cpp/issues/37
**State**: closed
**Created**: 2023-03-12T06:17:06+00:00
**Closed**: 2023-03-12T08:17:07+00:00
**Comments**: 2
**Labels**: build

### Description

I’m trying to compile main to play around with it and failing with error:

```
ld: symbol(s) not found for architecture arm64
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```

on macOS M1

trying to compile by running  `g++ main.cpp -o main -v -std=c++11`

anyone know what I'm missing?

---

## Issue #N/A: Eval bug: T5Encoder support broken

**Link**: https://github.com/ggml-org/llama.cpp/issues/12588
**State**: closed
**Created**: 2025-03-26T11:30:09+00:00
**Closed**: 2025-03-27T10:43:34+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

llama-cli --version
version: 4940 (fac63a3d)
built with Apple clang version 16.0.0 (clang-1600.0.26.6) for arm64-apple-darwin24.2.0

### Operating systems

Mac

### GGML backends

Metal

### Hardware

Mac mini M4

### Models

T5Encoder

### Problem description & steps to reproduce

Related to: https://github.com/HighDoping/Wan2.1/issues/2
T5Encoder support is broken after a recent code refactoring. 
The model support is commented out at https://github.com/ggml-org/llama.cpp/blob/5ed38b6852bd509d56acfdae54bceec8ab3cc396/src/llama-model.cpp#L11849-L11852

### First Bad Commit

https://github.com/ggml-org/llama.cpp/commit/e0dbec0bc6cd4b6230cda7a6ed1e9dac08d1600b

### Relevant log output

```shell
build: 4940 (fac63a3d) with Apple clang version 16.0.0 (clang-1600.0.26.6) for arm64-apple-darwin24.2.0
llama_model_load_from_file_impl: using device Metal (Apple M4) - 21845 MiB free
llama_model_loader: loaded meta data with 32 key-value pairs and 242 tensors from ../Wan2.1

[... truncated for brevity ...]

---

## Issue #N/A: [User] Insert summary of your issue or enhancement..

**Link**: https://github.com/ggml-org/llama.cpp/issues/789
**State**: closed
**Created**: 2023-04-05T19:59:06+00:00
**Closed**: 2023-04-06T00:29:02+00:00
**Comments**: 0

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `llama.cpp` to do.

# Current Behavior

Please provide a detailed written description of what `llama.cpp` did, instead. 

# Environment and Context 

Please prov

[... truncated for brevity ...]

---

## Issue #N/A: make -j LLAMA_CUBLAS=1 LLAMA_CUDA_F16=1 failed

**Link**: https://github.com/ggml-org/llama.cpp/issues/6194
**State**: closed
**Created**: 2024-03-21T09:02:39+00:00
**Closed**: 2024-03-21T12:59:54+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

make -j LLAMA_CUBLAS=1 LLAMA_CUDA_F16=1 failed to compile. Remove LLAMA_CUDA_F16=1 will fix.

```
ggml-cuda.cu(9456): error: identifier "cuda_pool_alloc" is undefined
      cuda_pool_alloc<half> src1_dfloat_a;
      ^

ggml-cuda.cu(9456): error: type name is not allowed
      cuda_pool_alloc<half> src1_dfloat_a;
                      ^

ggml-cuda.cu(9456): error: identifier "src1_dfloat_a" is undefined
      cuda_pool_alloc<half> src1_dfloat_a;
```



---

## Issue #N/A: Compile bug: error: passing 'const ggml_fp16_t *' (aka 'const unsigned short *') to parameter of type 'ggml_fp16_t *' (aka 'unsigned short *') discards qualifiers [-Werror,-Wincompatible-pointer-types-discards-qualifiers]

**Link**: https://github.com/ggml-org/llama.cpp/issues/10955
**State**: closed
**Created**: 2024-12-23T09:46:26+00:00
**Closed**: 2024-12-23T19:25:53+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Git commit

4381

### Operating systems

BSD

### GGML backends

CPU

### Problem description & steps to reproduce

Build fails.

Triggered by: ci/run.sh

### First Bad Commit

_No response_

### Relevant log output

```shell
/usr/ports/misc/llama-cpp/work/llama.cpp-b4381/ggml/src/ggml-cpu/ggml-cpu.c:1602:39: error: passing 'const ggml_fp16_t *' (aka 'const unsigned short *') to parameter of type 'ggml_fp16_t *' (aka 'unsigned short *') discards qualifiers [-Werror,-Wincompatible-pointer-types-discards-qualifiers]
 1602 |             ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
      |                     ~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~
/usr/ports/misc/llama-cpp/work/llama.cpp-b4381/ggml/src/ggml-cpu/ggml-cpu.c:1024:55: note: expanded from macro 'GGML_F16_VEC_LOAD'
 1024 | #define GGML_F16_VEC_LOAD(p, i)      GGML_F32Cx4_LOAD(p)
      |                                      ~~~~~~~~~~~~~~~~~^~
/usr/ports/misc/llama-cpp/work/llama.cpp-b4381/ggml/src/ggml

[... truncated for brevity ...]

---

## Issue #N/A: Smarter slot handling

**Link**: https://github.com/ggml-org/llama.cpp/issues/5737
**State**: closed
**Created**: 2024-02-26T16:42:18+00:00
**Closed**: 2024-02-27T15:17:39+00:00
**Comments**: 11

### Description

The current system of available slots with -np is frustrating in terms of how it forces one to only allow queries of greatly reduced max token count.  For example, if you have a context length of 16k, if you want four slots, each will only be 4k, and you can no longer run any 16k queries at all without them being heavily truncated.

While a partial solution would be to allow the operator to specify the numbers of token in each slot so that they could at least leave one high-token-count slot, an ideal solution would be to have the server be adaptive - to look at what's in the queue, and using a combination of how long each query has been waiting and how well different queries could be packed into the max context length, determine which to run and how many slots to use of what size.

While I wouldn't be an ideal person to write the slot-handling side of things, I'd be more than happy to write the queueing mechanism for you if this were of interest.  I would just need to know what sor

[... truncated for brevity ...]

---

## Issue #N/A: Builds after May 10 (master-cf348a6) not working at all

**Link**: https://github.com/ggml-org/llama.cpp/issues/1577
**State**: closed
**Created**: 2023-05-23T21:40:46+00:00
**Closed**: 2023-05-23T23:14:32+00:00
**Comments**: 1

### Description

I am running this command: `main.exe -m ./models/ggml-alpaca-7b-q4.bin --color -f ./prompts/alpaca.txt --ctx_size 2048 -n -1 -ins -b 256 --top_k 10000 --temp 0.2 --repeat_penalty 1.1 -t 7`

In version master-cf348a6, it successfully loads the model.

In version master-7d87381, I see this on the console:
```
main: build = 585 (7d87381)
main: seed  = 1684877783
llama.cpp: loading model from ./models/ggml-alpaca-7b-q4.bin
```
Then the program abruptly exists.  If I run it in a debugger, I see that it encountered a "CPP-EH-EXCEPTION".

# Environment and Context

Windows 11
16GB of RAM
AMD Ryzen 7 5800H


Edit:

Dies here:
```
    void read_raw(void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Weird output from CodeQwen converted from safetensors and unrecognized BPE pre-tokenizer for CodeQwen

**Link**: https://github.com/ggml-org/llama.cpp/issues/7939
**State**: closed
**Created**: 2024-06-14T15:22:09+00:00
**Closed**: 2024-06-15T06:48:49+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

When trying to convert CodeQwen safetensors into GGUF, I get this error:
```
python3.10 convert-hf-to-gguf.py ./models--Qwen--CodeQwen1.5-7B-Chat/snapshots/7b0cc3380fe815e6f08fe2f80c03e05a8b1883d8/ --outfile test.gguf 

INFO:hf-to-gguf:Loading model: 7b0cc3380fe815e6f08fe2f80c03e05a8b1883d8
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only
INFO:hf-to-gguf:Set model parameters
INFO:hf-to-gguf:gguf: context length = 65536
INFO:hf-to-gguf:gguf: embedding length = 4096
INFO:hf-to-gguf:gguf: feed forward length = 13440
INFO:hf-to-gguf:gguf: head count = 32
INFO:hf-to-gguf:gguf: key-value head count = 4
INFO:hf-to-gguf:gguf: rope theta = 1000000
INFO:hf-to-gguf:gguf: rms norm epsilon = 1e-05
INFO:hf-to-gguf:gguf: file type = 1
INFO:hf-to-gguf:Set model tokenizer
WARNING:hf-to-gguf:

WARNING:hf-to-gguf:**************************************************************************************
WARNING:hf-to-gguf:** WARNING: The BPE pre-token

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: llama-server -hf nomic-ai/nomic-embed-text-v2-moe-GGUF --embeddings , broken on latest version

**Link**: https://github.com/ggml-org/llama.cpp/issues/14021
**State**: closed
**Created**: 2025-06-05T06:20:34+00:00
**Closed**: 2025-06-05T07:29:19+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

`llama-server -hf nomic-ai/nomic-embed-text-v2-moe-GGUF:Q4_K_M --embeddings`
this version is OK
```
llama-server --version
version: 5569 (e57bb87c)
built with cc (GCC) 11.5.0 20240719 (Red Hat 11.5.0-2.0.1) for x86_64-redhat-linux

```

All subsequent versions include latest version have issues.

### Operating systems

Linux

### GGML backends

CPU

### Hardware

INTEL(R) XEON(R) GOLD 6530

### Models

_No response_

### Problem description & steps to reproduce

`llama-server -hf nomic-ai/nomic-embed-text-v2-moe-GGUF:Q4_K_M --embeddings`
fail load model

### First Bad Commit

_No response_

### Relevant log output

```shell
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 321.66 MiB (5.68 BPW)
load: model vocab missing newline token, using special_pad_id instead
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: special tokens cache size = 4
load: token to piece c

[... truncated for brevity ...]

---

## Issue #N/A: Download ggml-alpaca-7b-q4.bin failed CHECKSUM

**Link**: https://github.com/ggml-org/llama.cpp/issues/410
**State**: closed
**Created**: 2023-03-22T21:31:37+00:00
**Closed**: 2023-03-23T09:22:24+00:00
**Comments**: 15
**Labels**: model

### Description

This may well be the end server issue. I tried several times with no luck, just wonder if people have seen this. 
I tried all 3 curl commands. 



---

## Issue #N/A: Segfault when submitting image to ggml-org/Qwen2.5-VL-7B-Instruct-GGUF

**Link**: https://github.com/ggml-org/llama.cpp/issues/13467
**State**: closed
**Created**: 2025-05-12T05:25:09+00:00
**Closed**: 2025-05-12T13:06:52+00:00
**Comments**: 0

### Description

I am getting the following error/segfault
when submitting an image to ggml-org/Qwen2.5-VL-7B-Instruct-GGUF
using llama-server and the Python llm package as a client.

The same image is processed perfectly fine with
ggml-org/gemma-3-4b-it-GGUF
and
ggml-org/SmolVLM2-2.2B-Instruct-GGUF.

This is the output of "identify" on the image file:
neocube-one-layer-pattern.jpg JPEG 2592x1944 2592x1944+0+0 8-bit sRGB 858245B 0.000u 0:00.000

And here is the output I get from llama-server before it segfaults.
Note that it wants to allocate 44GB RAM, which is likely an error somewhere.

```
slot launch_slot_: id  0 | task 0 | processing task
slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 4096, n_keep = 0, n_prompt_tokens = 11
slot update_slots: id  0 | task 0 | kv cache rm [0, end)
slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 4, n_tokens = 4, progress = 0.363636
encoding image or slice...
slot update_slots: id  0 | task 0 | kv cache rm [4, end)
srv  process_c

[... truncated for brevity ...]

---

## Issue #N/A: Missing "scripts" module, but scripts appears to be a folder with some code.

**Link**: https://github.com/ggml-org/llama.cpp/issues/3239
**State**: closed
**Created**: 2023-09-18T00:27:05+00:00
**Closed**: 2023-09-18T02:31:52+00:00
**Comments**: 0

### Description

I just updated my copy from the repo today, and everything stopped working.
Ubuntu 20.04
gcc/g++ are 9.x.x

(CodeLlama) developer@ai:~/llama.cpp$ make clean
Traceback (most recent call last):
  File "/home/developer/mambaforge/envs/CodeLlama/bin/make", line 5, in <module>
    from scripts.proto import main
ModuleNotFoundError: No module named 'scripts.proto'
(CodeLlama) developer@ai:~/llama.cpp$ vi Makefile
(CodeLlama) developer@ai:~/llama.cpp$ LLAMA_CUBLAS=1 make
Traceback (most recent call last):
  File "/home/developer/mambaforge/envs/CodeLlama/bin/make", line 5, in <module>
    from scripts.proto import main
ModuleNotFoundError: No module named 'scripts.proto'
(CodeLlama) developer@ai:~/llama.cpp$


---

