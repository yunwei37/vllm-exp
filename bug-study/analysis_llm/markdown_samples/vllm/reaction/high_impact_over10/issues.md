# high_impact_over10 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 26

### Label Distribution

- feature request: 9 issues
- stale: 9 issues
- RFC: 5 issues
- bug: 5 issues
- new-model: 4 issues
- x86-cpu: 1 issues
- unstale: 1 issues
- startup-ux: 1 issues
- good first issue: 1 issues

---

## Issue #N/A: vLLM running on a Ray Cluster Hanging on Initializing

**Link**: https://github.com/vllm-project/vllm/issues/2826
**State**: closed
**Created**: 2024-02-09T14:41:23+00:00
**Closed**: 2024-02-27T01:33:39+00:00
**Comments**: 12

### Description

It isn't clear what is at fault here.  Whether it be vLLM or Ray.

There is a thread here on the ray forums that outlines the issue, it is 16 days old, there is no reply to it.

https://discuss.ray.io/t/running-vllm-script-on-multi-node-cluster/13533

Taking from that thread, but this is identical for me.

```
2024-01-24 13:57:17,308 INFO worker.py:1540 – Connecting to existing Ray cluster at address: HOST_IP_ADDRESS…
2024-01-24 13:57:17,317 INFO worker.py:1715 – Connected to Ray cluster. View the dashboard at 127.0.0.1:8265
INFO 01-24 13:57:39 llm_engine.py:70] Initializing an LLM engine with config: model=‘mistralai/Mistral-7B-Instruct-v0.2’, tokenizer=‘mistralai/Mistral-7B-Instruct-v0.2’, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=auto, tensor_parallel_size=2, quantization=None, enforce_eager=False, seed=0)

But after that it hangs, and eventually quits.
`

[... truncated for brevity ...]

---

## Issue #N/A: can model  Qwen/Qwen-VL-Chat work well?

**Link**: https://github.com/vllm-project/vllm/issues/962
**State**: closed
**Created**: 2023-09-06T10:18:59+00:00
**Closed**: 2024-09-05T12:48:12+00:00
**Comments**: 11
**Labels**: new-model

### Description

when i use Qwen/Qwen-VL-Chat  I do not know why!

throw a error 

`Traceback (most recent call last):
  File "test.py", line 20, in <module>
    model = LLM(model=model_path, tokenizer=model_path,tokenizer_mode='slow',tensor_parallel_size=1,trust_remote_code=True)
  File "/usr/local/miniconda3/lib/python3.8/site-packages/vllm/entrypoints/llm.py", line 66, in __init__
    self.llm_engine = LLMEngine.from_engine_args(engine_args)
  File "/usr/local/miniconda3/lib/python3.8/site-packages/vllm/engine/llm_engine.py", line 220, in from_engine_args
    engine = cls(*engine_configs,
  File "/usr/local/miniconda3/lib/python3.8/site-packages/vllm/engine/llm_engine.py", line 101, in __init__
    self._init_workers(distributed_init_method)
  File "/usr/local/miniconda3/lib/python3.8/site-packages/vllm/engine/llm_engine.py", line 133, in _init_workers
    self._run_workers(
  File "/usr/local/miniconda3/lib/python3.8/site-packages/vllm/engine/llm_engine.py", line 470, in _run_workers

[... truncated for brevity ...]

---

## Issue #N/A: [RFC] Initial Support for CPUs

**Link**: https://github.com/vllm-project/vllm/issues/3654
**State**: closed
**Created**: 2024-03-27T07:45:25+00:00
**Closed**: 2025-01-14T16:19:23+00:00
**Comments**: 11
**Labels**: RFC, x86-cpu, unstale

### Description

## Progress

- [ ] Integrate CPU executor to support the basic model inference (BF16/FP32) without TP. 
  - #3634 
  - #3824 
  - #4113 
  - #4971 
  - #5452 
  - #5446 
- [ ] Support FP16 model inference.
- [x] Support TP inference for multiple CPU sockets inside the same node. 
  - #6008 
  - #6125 
- [ ] Support model and KV cache quantization.
  - #5492 
  - #7257 

## Features

The CPU executor plans to support the following features:

- Basic models of vLLM with FP16/BF16/FP32, except MoE models
- Tensor-parallel model inference based on Ray
- AWQ quantization, 8-bit KVCache Quantization
- Others

## Design

Our target is seamless porting vLLM to CPU devices and sharing most of vLLM core components (e.g., **schedular**, **cache management**, **model definitions**, **Megatron-style model partitioning**, ...). 

The CPU executor will depend on Pytorch CPU and leverage optimized kernels and features from [intel-extension-for-pytorch](https://github.com/

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: support tool and reasoning together

**Link**: https://github.com/vllm-project/vllm/issues/14429
**State**: open
**Created**: 2025-03-07T10:19:33+00:00
**Comments**: 17
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

For now `--enable-auto-tool-choice` and `--enable-reasoning` can't enable together, with the following errors:
```
# vllm serve /Qwen/QwQ-32B/ --served-model-name QwQ-32B --gpu-memory-utilization 0.97 --tensor-parallel-size 8  --max-model-len 32768  --enable-reasoning --reasoning-parser deepseek_r1  --enable-auto-tool-choice --tool-call-parser hermes
INFO 03-07 18:14:44 [__init__.py:207] Automatically detected platform cuda.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/usr/local/lib/python3.12/site-packages/vllm/entrypoints/cli/main.py", line 70, in main
    cmds[args.subparser].validate(args)
  File "/usr/local/lib/python3.12/site-packages/vllm/entrypoints/cli/serve.py", line 36, in validate
    validate_parsed_serve_args(args)
  File "/usr/local/lib/python3.12/site-packages/vllm/entrypoints/openai/cli_args.py", line 285, in validate_parsed_serve_args
    

[... truncated for brevity ...]

---

## Issue #N/A: [Feature Request]: Support data_parallel_size in offline inference mode

**Link**: https://github.com/vllm-project/vllm/issues/16588
**State**: open
**Created**: 2025-04-14T11:59:51+00:00
**Comments**: 9
**Labels**: feature request

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 04-14 19:54:10 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: CentOS Linux 7 (Core) (x86_64)
GCC version: (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2)
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.17

Python version: 3.12.3 | packaged by conda-forge | (main, Apr 15 2024, 18:38:13) [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-4.18.0-147.mt20200626.413.el8_1.x86_64-x86_64-with-glibc2.17
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB
GPU 2: NVIDIA A100-SXM4-80GB
GPU 3: NVIDIA A100-SXM4-80GB

Nvidia driver version: 470.103.01
cuDNN version: Prob

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: allenai/Molmo-7B-0-0924 VisionLM

**Link**: https://github.com/vllm-project/vllm/issues/8808
**State**: closed
**Created**: 2024-09-25T16:34:48+00:00
**Closed**: 2024-10-14T14:56:25+00:00
**Comments**: 17
**Labels**: new-model

### Description

### The model to consider.

https://huggingface.co/allenai/Molmo-7B-O-0924
https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19

### The closest model vllm already supports.

Existing Olmo Models by AllenAi: `OLMoForCausalLM` and `OLMoEForCausalLM` are supported.

### What's your difficulty of supporting the model you want?

Molmo is a vision LM, so unlike the previous Olmo models by Allen AI, this model includes vision.

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Import FlashInfer: 3x faster PagedAttention than vLLM

**Link**: https://github.com/vllm-project/vllm/issues/2767
**State**: closed
**Created**: 2024-02-05T18:21:00+00:00
**Closed**: 2024-08-06T02:08:00+00:00
**Comments**: 2

### Description

It looks like vLLM could directly import the PagedAttention kernels from FlashInfer to support GQA. "*For batch GQA decoding attention, FlashInfer w/ Tensor Cores is 3x faster than vLLM PagaAttention when batch_size=64.*" @WoosukKwon 

https://github.com/flashinfer-ai/flashinfer/
https://flashinfer.ai/2024/02/02/introduce-flashinfer.html

![image](https://github.com/vllm-project/vllm/assets/27340033/48d40b10-a5d0-4ea3-9c9f-53cc8a7bca4a)


---

## Issue #N/A: [question] Does vllm support macos M1 or M2 chip?

**Link**: https://github.com/vllm-project/vllm/issues/1397
**State**: closed
**Created**: 2023-10-17T14:09:56+00:00
**Closed**: 2024-05-31T19:57:52+00:00
**Comments**: 5

### Description

[question] Does vllm support macos M1 or M2 chip? I see the codes just containing cuda?

---

## Issue #N/A: [RFC]: Scale the API server across multiple CPUs

**Link**: https://github.com/vllm-project/vllm/issues/12705
**State**: closed
**Created**: 2025-02-03T19:14:29+00:00
**Closed**: 2025-06-11T22:52:20+00:00
**Comments**: 12
**Labels**: RFC

### Description

### Motivation.

Currently, the API server runs in a single process, utilizing a single CPU for its work. As GPUs continue to get faster, it is important that we scale the API server to ensure that it is able to process requests fast enough to keep GPU resources fully utilized.

### Proposed Change.

From a high level, this proposal is to move from the API server being a single process to being a configurable pool of processes to ensure that a single CPU for the apiserver will not become a bottleneck in server utilization.

Design notes: https://docs.google.com/document/d/1Y2S011RKYkFKtrcz_MuEqEf3cRXORNGsVvMHCaqqc-k/edit?tab=t.0

### Feedback Period.

_No response_

### CC List.

@robertgshaw2-redhat @njhill 

### Any Other Things.

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of fre

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: Llama 3.2

**Link**: https://github.com/vllm-project/vllm/issues/8812
**State**: closed
**Created**: 2024-09-25T18:01:58+00:00
**Closed**: 2024-09-25T20:29:34+00:00
**Comments**: 0
**Labels**: new-model

### Description

### The model to consider.
- **Huggingface collection:** https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf

Highlighted model weights:
- **1B Instruct Model:** https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- **3B Instruct Model:** https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- **11B Instruct Model:** https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
- **90B Instruct Model:** https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct

### The closest model vllm already supports.

https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava_onevision.py

### What's your difficulty of supporting the model you want?

Yes, Llama 3.2 is multimodal with a different architecture than previous multimodal Llama models.

### Before submitting a new issue...

- [X] Make sure you already searched for relevant 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM MQLLMEngine Timeout - Json Schema 

**Link**: https://github.com/vllm-project/vllm/issues/9082
**State**: closed
**Created**: 2024-10-04T20:29:57+00:00
**Closed**: 2024-10-30T16:34:09+00:00
**Comments**: 7
**Labels**: bug

### Description

### Your current environment

<details>

<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` 
Collecting environment information...
PyTorch version: N/A
Is debug build: N/A
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.12.4 | packaged by Anaconda, Inc. | (main, Jun 18 2024, 15:12:24) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.10.225-213.878.amzn2.x86_64-x86_64-with-glibc2.35
Is CUDA available: N/A
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: GPU 0: NVIDIA A10G
Nvidia driver version: 550.90.12
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: N/A

CPU:
Architecture:                

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Deprecation and removal for `--engine-use-ray`

**Link**: https://github.com/vllm-project/vllm/issues/7045
**State**: closed
**Created**: 2024-08-01T20:17:35+00:00
**Closed**: 2024-08-14T16:44:28+00:00
**Comments**: 4
**Labels**: RFC

### Description

### Motivation.

In the `async_engine` code path, we have an option to launch the engine in a separate process using Ray

```python
        parser.add_argument('--engine-use-ray',
                            action='store_true',
                            help='Use Ray to start the LLM engine in a '
                            'separate process as the server process
```

Originally, the option make it possible to separate the server's Python overhead with the engine's main scheduler loop. 

However, few factors made this unused/less popular
* Ray is an optional component, and typically not used in single node environment.
* The serialization and rpc typically offset the theoretical performance gain
* There are typically other ways to isolate server and engine (through multiprocessing, threading, etc).
* Recently, we are separating this in server using lower overhead approaches #6883

### Proposed Change.

Deprecation of the flag with warning for one release. 
Removal o

[... truncated for brevity ...]

---

## Issue #N/A: GPTQ / Quantization support?

**Link**: https://github.com/vllm-project/vllm/issues/174
**State**: closed
**Created**: 2023-06-21T02:40:47+00:00
**Closed**: 2024-03-06T09:01:49+00:00
**Comments**: 19
**Labels**: feature request

### Description

Will vLLM support 4-bit GPTQ models?

---

## Issue #N/A: [Feature]: Support to use draft models with different vocabulary sizes for speculative decoding

**Link**: https://github.com/vllm-project/vllm/issues/7252
**State**: closed
**Created**: 2024-08-07T07:39:33+00:00
**Closed**: 2024-12-06T02:07:01+00:00
**Comments**: 6
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

In most open-source LLM families, models with different parameters use the same tokenizer and vocabulary. However, due to differences in GPU infrastructure during training, they might use different numbers of padding tokens, resulting in different `vocab_size` values.

For instance, the `vocab_size` of Qwen2's [1.5B version](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct/blob/main/config.json) is **151936**, while the [72B version](https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/config.json) is **152064**. These padding tokens are essentially [meaningless at inference time](https://github.com/QwenLM/Qwen2/issues/466), but when used for speculative decoding, vLLM raises an error due to the mismatch in vocabulary size.

Therefore, I propose adding an engine argument, such as `--disable-vocab-check-for-spec-decoding`, to allow the use of draft models with different vocabulary sizes upon user confirmation.

### Alternatives

Addi

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Add opentelemetry tracing for vLLM start up phases

**Link**: https://github.com/vllm-project/vllm/issues/19318
**State**: open
**Created**: 2025-06-07T16:10:23+00:00
**Comments**: 2
**Labels**: feature request, startup-ux

### Description

### 🚀 The feature, motivation and pitch

This FR asks for tracing through vLLM cold starts. This would include key phases, as trace spans, leading up to the FastAPI HTTP server is up and running. #17794 is related but asks for tracing requests.

Why would this be useful?

* To facilitate cold start optimizations, both for vLLM users and contributors.
  This is important for quick auto scaling of inference workloads in cloud
  environments.
* Users may want to tweak vLLM settings based on which phase is contributig to
  high latency, e.g. changing how the model is loaded using `--load-format
  runai_streamer`.
* Contributors interested in performance optimization need this data to know
  which area to focus on and the visual phase breakdown provided by traces is
  much easier to interpret quickly than logs. This is how I noticed #19317.

The set of key spans and their attributes could be iterated on over time but I
think it'd be interesting to include at least

* Python import time (whi

[... truncated for brevity ...]

---

## Issue #N/A: TypeError: 'NoneType' object is not callable

**Link**: https://github.com/vllm-project/vllm/issues/3057
**State**: closed
**Created**: 2024-02-27T12:49:04+00:00
**Closed**: 2024-11-29T02:08:02+00:00
**Comments**: 7
**Labels**: stale

### Description

when I test for mutil-gpu with llama2-70b, run `vllm/examples/offline_inference.py` , use params `enforce_eager=False`, the result can output, but it occur some error
```
Prompt: 'Hello, my name is', Generated text: ' Dustin Nelson and I’m going to be posting articles and my thoughts' 
Prompt: 'The president of the United States is', Generated text: ' one of the most powerful people in the world, as the leader of the only' 
Prompt: 'The capital of France is', Generated text: ' one of the world’s leading cities in terms of art, fashion, food' 
Prompt: 'The future of AI is', Generated text: ' neither utopian nor apocalyptic—it’s both.\n' 
Exception ignored in: <function TCPStore.__del__ at 0x7f930d38e8c0>
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/cupyx/distributed/_store.py", line 59, in __del__
  File "/usr/local/lib/python3.10/dist-packages/cupyx/distributed/_store.py", line 109, in stop
  File "/usr/local/lib/python3.10/dist-packages/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: MistralTokenizer not working when using Mistral Small 3.1 in HF format

**Link**: https://github.com/vllm-project/vllm/issues/16292
**State**: closed
**Created**: 2025-04-08T22:30:02+00:00
**Closed**: 2025-04-29T02:53:45+00:00
**Comments**: 10
**Labels**: bug

### Description

### Your current environment

vLLM v0.8.3

### 🐛 Describe the bug

vLLM v0.8.3 fails on start when using Mistral Small 3.1 in HF format and the mistral tokenizer (required for proper function calling parsing)

Launch command : 
`docker run --runtime nvidia --gpus all -v /path/models:/models -e HF_CACHE=/models --ipc=host vllm/vllm-openai:v0.8.3 --model /models/Mistral-Small-3.1-24B-Instruct-2503-FP8-KV --download-dir /models --kv-cache-dtype fp8 --limit_mm_per_prompt 'image=10' --max-model-len 65536 --enable-auto-tool-choice  --tool-call-parser mistral --tokenizer-mode mistral`

<details>
<summary>Error when loading vLLM</summary>

```text
INFO 04-08 15:05:42 [gpu_model_runner.py:1542] Encoder cache will be initialized with a budget of 8192 tokens, and profiled with 3 image items of the maximum feature size.
ERROR 04-08 15:05:44 [core.py:390] EngineCore hit an exception: Traceback (most recent call last):
ERROR 04-08 15:05:44 [core.py:390]   File "/usr/local/lib/python3.12/dist-package

[... truncated for brevity ...]

---

## Issue #N/A: [new feature] flash decoding ++

**Link**: https://github.com/vllm-project/vllm/issues/1568
**State**: closed
**Created**: 2023-11-05T12:35:56+00:00
**Closed**: 2025-03-11T13:51:37+00:00
**Comments**: 10
**Labels**: feature request, stale

### Description

Recently flashdecoding++ is introduced by below paper. It could boost the decoding efficiency. Would you like to implement that?
https://arxiv.org/pdf/2311.01282.pdf
Thank you in advance. 

---

## Issue #N/A: Phi 1.5 support

**Link**: https://github.com/vllm-project/vllm/issues/1167
**State**: closed
**Created**: 2023-09-24T18:08:58+00:00
**Closed**: 2023-11-16T22:28:40+00:00
**Comments**: 15
**Labels**: new-model

### Description

Phi 1.5 is a new model from Microsoft, supporting this model would be extremely usefull.

A detailed list of info of phi 1.5 can be found here : [https://huggingface.co/microsoft/phi-1_5](url)

Its basically supporting MixFormerSequentialConfig .
The phi 1.5 has weird features, also 4 bit support would be great !! (and not only on gpu, but cpu also please, this model size should work ok on cpu)


---

## Issue #N/A: [Bug]: TRACKING ISSUE: `AsyncEngineDeadError`

**Link**: https://github.com/vllm-project/vllm/issues/5901
**State**: open
**Created**: 2024-06-27T11:49:38+00:00
**Comments**: 23
**Labels**: bug, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### 🐛 Describe the bug

Recently, we have seen reports of `AsyncEngineDeadError`, including:

- [ ] #5060
- [x] #2000
- [x] #3310
- [x] #3839
- [x] #4000
- [x] #4135
- [x] #4293
- [x] #5443
- [x] #5732
- [x] #5822
- [x] #6190 
- [x] #6208
- [x] #6361
- [x] #6421
- [ ] #6614
- [x] #6790
- [x] #6969
- [x] #7356

If you see something like the following, please report here:

```bash
2024-06-25 12:27:29.905   File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/openai/api_server.py", line 84, in health
2024-06-25 12:27:29.905     await openai_serving_chat.engine.check_health()
2024-06-25 12:27:29.905   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 839, in check_health
2024-06-25 12:27:29.905     raise AsyncEngineDeadError("Background loop is stopped.")
2024-06-25 12:27:29.905 vllm.engine.async_llm_engine.AsyncEngineDeadError:

[... truncated for brevity ...]

---

## Issue #N/A: Support for grammar

**Link**: https://github.com/vllm-project/vllm/issues/1229
**State**: closed
**Created**: 2023-09-29T17:41:17+00:00
**Closed**: 2024-03-16T20:35:29+00:00
**Comments**: 8

### Description

It would be highly beneficial if the library could incorporate support for Grammar and GBNF files.
https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

---

## Issue #N/A: Loading Model through Multi-Node Ray Cluster Fails

**Link**: https://github.com/vllm-project/vllm/issues/881
**State**: closed
**Created**: 2023-08-25T23:14:00+00:00
**Closed**: 2024-05-31T19:36:15+00:00
**Comments**: 7

### Description

## Problem Description

I'm trying to spin up the VLLM API server inside a docker container (that has vllm and all requirements installed) on the head node of a ray cluster (the cluster contains 4 nodes and has access to 4 T4 GPUs) with the following command in the container:

`python3 -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8000 --model /home/model --tensor-parallel-size 4 --swap-space 2
`

I get back the following error (NOTE: I x'd out the IP):

```
python3 -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8000 --model /home/model --tensor-parallel-size 4 --swap-space^Me 2
^[[?2004l^M2023-08-25 22:29:16,736      INFO worker.py:1313 -- Using address ray://xxx.xxx.xxx.xx:10001 set in the environment variable RAY_ADDRESS
2023-08-25 22:29:16,737 INFO client_builder.py:237 -- Passing the following kwargs to ray.init() on the server: ignore_reinit_error
INFO 08-25 22:29:19 llm_engine.py:70] Initializing an LLM engine with config: model='/home/model', tokenizer

[... truncated for brevity ...]

---

## Issue #N/A: Speculative Streaming: Fast LLM Inference without Auxiliary Models

**Link**: https://github.com/vllm-project/vllm/issues/2943
**State**: closed
**Created**: 2024-02-21T00:17:57+00:00
**Closed**: 2024-11-30T02:02:29+00:00
**Comments**: 3
**Labels**: feature request, stale

### Description

This might be of interest: https://arxiv.org/pdf/2402.11131.pdf

---

## Issue #N/A: Error installing on windows

**Link**: https://github.com/vllm-project/vllm/issues/2309
**State**: closed
**Created**: 2023-12-31T01:32:15+00:00
**Closed**: 2024-04-18T14:00:12+00:00
**Comments**: 15

### Description

NameError: name 'nvcc_cuda_version' is not defined. Did you mean: 'cuda_version'?

This was a simple fix I defined the cuda version in line 268 of setup .py and installed with 'pip install .'

so change line 268
from:         cuda_version = str(nvcc_cuda_version)
to:         cuda_version = str(12.1)


If this problem is common amongst windows users you could add a precheck for os version, and if windows, allow user to set cuda version via prompt.


Although i still have a error building the wheels further down the line:

      copying vllm\transformers_utils\tokenizers\__init__.py -> build\lib.win-amd64-cpython-311\vllm\transformers_utils\tokenizers
      copying vllm\py.typed -> build\lib.win-amd64-cpython-311\vllm
      running build_ext
      C:\Users\PC\AppData\Local\Temp\pip-build-env-8j8g0uyh\overlay\Lib\site-packages\torch\utils\cpp_extension.py:383: UserWarning: Error checking compiler version for cl: [WinError 2] The system cannot find the file specified
    

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Dynamic KV Cache compression based on  vLLM framework

**Link**: https://github.com/vllm-project/vllm/issues/10942
**State**: closed
**Created**: 2024-12-06T03:40:05+00:00
**Closed**: 2025-05-10T02:06:52+00:00
**Comments**: 4
**Labels**: RFC, stale

### Description

### Motivation.

# KV Sparsity and Model Compression

By reviewing recent academic papers from the past year in the field of KV sparsity (H2O, SnapKV, PyramidKV), we apply KV sparsity to different layers of the model. By employing a pruning strategy, we eliminate KV pairs with lower scores while retaining those with higher scores and closer proximity. This approach reduces memory usage, as well as computational and I/O overhead, ultimately leading to accelerated inference.

## Experiments

### Baselines and Settings

We run all KV-Compress experiments using our vLLM integration forked from v0.6.2, running in CUDA graph mode with a block size of 16. For all RTX 4090 / Llama-3.1-8B-Instruct experiments, we use the default GPU memory utilization of 0.9 and set the `maxmodel-length` to 32k.  
We evaluate our compression on Llama-3.1-8B-Instruct, comparing performance against the following baseline methods introduced in prior work:

- vLLM-0.6.2
- Novita AI, Pyramid KV Cache com

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: support reward model API

**Link**: https://github.com/vllm-project/vllm/issues/6620
**State**: closed
**Created**: 2024-07-21T10:36:57+00:00
**Closed**: 2024-11-24T02:07:44+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

Does VLLM support rapid deployment of RM services? 
Or convenient custom APIs? It seems that currently there are only chat/completion/embedding APIs. As a newcomer to inference acceleration, any help would be beneficial.
We want to use vllm to accelerate RM API for the remote RM feature of OpenRLHF. https://github.com/OpenRLHF/OpenRLHF/pull/361/

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [Bug]: Docker deployment returns zmq.error.ZMQError: Operation not supported

**Link**: https://github.com/vllm-project/vllm/issues/10856
**State**: closed
**Created**: 2024-12-03T09:51:54+00:00
**Closed**: 2025-06-15T02:15:44+00:00
**Comments**: 24
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 10.5.0-1ubuntu1~22.04) 10.5.0
Clang version: Could not collect
CMake version: version 3.30.2
Libc version: glibc-2.35

Python version: 3.10.4 (main, Mar 31 2022, 08:41:55) [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-6.2.0-33-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 11.8.89
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A40
GPU 1: NVIDIA A40
GPU 2: NVIDIA A40
GPU 3: NVIDIA A40

Nvidia driver version: 535.86.05
cuDNN version: Probably one of the following:
/usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudnn.so.8.9.5
/usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudnn_adv_

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Add support for `logit_bias`

**Link**: https://github.com/vllm-project/vllm/issues/379
**State**: closed
**Created**: 2023-07-06T11:42:15+00:00
**Closed**: 2024-03-19T22:45:10+00:00
**Comments**: 5
**Labels**: good first issue, feature request

### Description

Support just landed in `transformers` lib for this. See:
- [`SequenceBiasLogitsProcessor`](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.SequenceBiasLogitsProcessor)
- [Corresponding `GenerationConfig`](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.sequence_bias)
- [discussion](https://github.com/huggingface/transformers/issues/22168#issuecomment-1477998997)

---

## Issue #N/A: [Bug]:  ray cluster Segmentation fault

**Link**: https://github.com/vllm-project/vllm/issues/6106
**State**: closed
**Created**: 2024-07-03T14:47:23+00:00
**Closed**: 2024-11-24T02:08:44+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.6
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-113-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: Tesla V100-SXM2-32GB
GPU 1: Tesla V100-SXM2-32GB
GPU 2: Tesla V100-SXM2-32GB
GPU 3: Tesla V100-SXM2-32GB
GPU 4: Tesla V100-SXM2-32GB
GPU 5: Tesla V100-SXM2-32GB
GPU 6: Tesla V100-SXM2-32GB
GPU 7: Tesla V100-SXM2-32GB

Nvidia driver version: 535.183.01
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-g

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Reimplement and separate beam search on top of vLLM core

**Link**: https://github.com/vllm-project/vllm/issues/8306
**State**: closed
**Created**: 2024-09-09T20:17:13+00:00
**Closed**: 2024-10-07T05:47:05+00:00
**Comments**: 21
**Labels**: RFC

### Description

### Motivation.

A rework of https://github.com/vllm-project/vllm/issues/6226 

After discussing further with the community, we find that the common use case for beam search is: 
1. throughput oriented
2. mainly offline batch inference
3. use one beam search parameter for all the prompts in the batch

After discussing with many contributors, we find:

because beam search is a **search** algorithm, it conflicts with all the rest **sampling** algorithm. As a result, many features in vllm already directly assert beam search is not used, e.g.

https://github.com/vllm-project/vllm/blob/6e36f4fa6ce64619b9ea94c88a157f5783a63a65/vllm/spec_decode/batch_expansion.py#L303-L305

https://github.com/vllm-project/vllm/blob/6e36f4fa6ce64619b9ea94c88a157f5783a63a65/vllm/engine/output_processor/multi_step.py#L100-L103

**keeping beam-search as-is in the codebase, will not benefit current beam search user, as no optimization will target at better beam search performance. What's worse, very

[... truncated for brevity ...]

---

