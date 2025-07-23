# references_prs - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 8
- Closed Issues: 22

### Label Distribution

- bug: 10 issues
- feature request: 6 issues
- RFC: 6 issues
- stale: 6 issues
- good first issue: 3 issues
- unstale: 2 issues
- misc: 2 issues
- v1: 2 issues
- installation: 2 issues
- startup-ux: 1 issues

---

## Issue #N/A: [Feature]: Return hidden states (in progress?)

**Link**: https://github.com/vllm-project/vllm/issues/6165
**State**: open
**Created**: 2024-07-06T01:26:10+00:00
**Comments**: 19
**Labels**: feature request, unstale

### Description

### 🚀 The feature, motivation and pitch

I know this feature request sort of already exists: https://github.com/vllm-project/vllm/issues/5950
(and older, semi related requests) https://github.com/vllm-project/vllm/issues/3594 https://github.com/vllm-project/vllm/issues/1857

This is a similar pitch but I am creating a new issue as I noticed newer developments in the codebase. The pitch is to support returning hidden states when generating sequences. This enables many potential behaviors such as output classification, guardrails, etc. Whereas #5950 suggested a different step for embedding, I would suggest building it in as an option to EngineArgs or as an option that can be passed in with each generation request. 

I see that in `v0.5.1` there is already some new code in `ModelDriverBase` to support `return_hidden_states`. However, I don't see that supported yet in the LLM engine yet (not an input to `EngineArgs`). Basically, it seems like this feature is under development. I am 

[... truncated for brevity ...]

---

## Issue #N/A: [CI] [Flaky test] distributed/test_shm_broadcast.py is flaky

**Link**: https://github.com/vllm-project/vllm/issues/5848
**State**: closed
**Created**: 2024-06-25T23:22:24+00:00
**Closed**: 2024-06-26T04:56:04+00:00
**Comments**: 3
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

Distributed comm ops test failed with below stacktrace. [Buildkite](https://buildkite.com/vllm/ci-aws/builds/2539#01904f5d-0bfd-4a7a-96b2-cdc7f8b60b09)

```
[2024-06-25T12:58:33Z] distributed/test_shm_broadcast.py:72:
--
  | [2024-06-25T12:58:33Z] _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  | [2024-06-25T12:58:33Z]
  | [2024-06-25T12:58:33Z] fn = <function worker_fn_wrapper.<locals>.wrapped_fn at 0x7f8cc92afa30>
  | [2024-06-25T12:58:33Z] world_size = 4
  | [2024-06-25T12:58:33Z]
  | [2024-06-25T12:58:33Z]     def distributed_run(fn, world_size):
  | [2024-06-25T12:58:33Z]         number_of_processes = world_size
  | [2024-06-25T12:58:33Z]         processes = []
  | [2024-06-25T12:58:33Z]         for i in range(number_of_processes):
  | [2024-06-25T12:58:33Z]             env = {}
  | [2024-06-25T12:58:33Z]             env['RANK'] = str(i)
  | [2024-06-25T12:58:33Z]             env['LOCAL

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Prompt logprobs + APC compatibility

**Link**: https://github.com/vllm-project/vllm/issues/13409
**State**: closed
**Created**: 2025-02-17T15:33:22+00:00
**Closed**: 2025-02-17T17:27:11+00:00
**Comments**: 0
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

[#9880](https://github.com/vllm-project/vllm/pull/9880) adds sample and prompt logprobs support, however prompt logprobs currently require the server to be instantiated with `--no-enable-prefix-caching`; otherwise, a request with `prompt_logprobs=true` will cause the request to fail with the message "Prefix caching with prompt logprobs not yet supported on VLLM V1."

The challenge of using prompt logprobs alongside APC is how to recover the topk prompt logprobs from an APC cache hit. The existing APC implementation does not cache prompt logprobs; upon a cache hit, cached blocks are treated as "computed" & no prompt logprobs are available for the computed blocks.

### Alternatives

A few possible solutions:
* **Use APC cached KVs to recompute prompt logprobs if a request with `prompt_logprobs=true` triggers an APC cache hit.** This requires model code and `model_executor` code to support re-running prefill using cached KVs.
* **Cache prompt logpr

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Could not `pip install vllm` inside dockerfile after certain commit in `main` branch

**Link**: https://github.com/vllm-project/vllm/issues/9226
**State**: closed
**Created**: 2024-10-10T05:47:12+00:00
**Closed**: 2024-10-11T19:57:40+00:00
**Comments**: 6
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.4.0
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.26.4
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.25-051525-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090

Nvidia driver version: 545.23.08
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU op-mode(s)

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Lazy CUDA Graph capture

**Link**: https://github.com/vllm-project/vllm/issues/20098
**State**: open
**Created**: 2025-06-25T21:27:51+00:00
**Comments**: 11
**Labels**: RFC, startup-ux

### Description

### Motivation.

Currently vLLM captures cudagraphs as part of the engine initialization significantly slowing down vLLM startup time. By default, vLLM captures 66 graphs, which depending on model size and GPU type, can take more than 10s. This is not great UX (see #19824 for details).

In addition, It's most unlikely that all 66 graphs are actually needed, wasting both time and space.  

### Proposed Change.

We propose to capture cudagraphs lazily. Instead of performing dummy runs during the engine initialization phase, the idea is to do those runs somewhere in the CUDA piecewise backend, and only for the current runtime shape if not cached already.

Exact implementation needs to be worked out.

### Feedback Period.

one week

### CC List.

@ProExpertProg @aarnphm @charlesfrye  

### Any Other Things.

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documenta

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Refactor Worker and ModelRunner to consolidate control plane communication

**Link**: https://github.com/vllm-project/vllm/issues/5552
**State**: closed
**Created**: 2024-06-14T18:55:42+00:00
**Closed**: 2024-06-27T20:30:39+00:00
**Comments**: 13
**Labels**: RFC

### Description

### Motivation.

Currently, both the Worker and the ModelRunner classes contain multi-GPU control plane communication code, i.e. `broadcast_tensor_dict` calls. They look something like this:

```python
class Worker:
  def execute_model(self, execute_model_req=None):
    # Do some broadcast here.
    ...
    return self.model_runner.execute_model(execute_model_req)

class ModelRunner:
  def execute_model(self, execute_model_req=None):
    # Do some more broadcast here.
    ...
    return model_executable(...)
```

Because the ModelRunner class contains both model execution code and multi-GPU control plane communication code, it makes it difficult to improve upon the performance:
- Cannot swap out the control plane mechanism, e.g., using NCCL vs CPU-based serialization to move the inputs from the LLMEngine to the Workers
- Cannot switch to an SPMD design, where the rank 0 worker is moved off of the driver and executes the same code as the rest of the workers
- Diffic

[... truncated for brevity ...]

---

## Issue #N/A: [SpecDecode] Support EAGLE in V1

**Link**: https://github.com/vllm-project/vllm/issues/15901
**State**: open
**Created**: 2025-04-01T19:45:13+00:00
**Comments**: 21
**Labels**: speculative-decoding, v1

### Description

- [x] 1. Correctly initializing and loading the EAGLE draft model
- [x] 2. Consider the lookahead slots in the KV cache manager
- [x] 3. Cache `draft_probs` inside the model runner and correctly feed it to the rejection sampler in the next step (temporarily workaround: #16899)
- [x] 4. Handle the edge cases like when the draft model generates beyond `max_pos_embeddings`
- [ ] 5. Handle the seeds correctly
- [ ] 6. Do E2E correctness and performance tests
- [x] 7. Support prefix caching. Eagle requires special handling because Eagle's i-th KV cache is coupled with the i+1-th token ID. (@LiuXiaoxuanPKU)
- [ ] 8. Properly handle the sampling parameters that are not (currently) compatible with spec decoding (e.g., min_p).
- [x] 9. Use CUDA graphs for draft model. (@luyuzhe111)
- [x] 10. Support Eagle 3 (https://github.com/vllm-project/vllm/pull/16937)

_Originally posted by @WoosukKwon in https://github.com/vllm-project/vllm/issues/15729#issuecomment-2765192455_
            

---

## Issue #N/A: [Feature]: Support EPLB for More MoE Models, e.g. Qwen 3, Llama 4

**Link**: https://github.com/vllm-project/vllm/issues/20468
**State**: open
**Created**: 2025-07-04T05:06:31+00:00
**Comments**: 20
**Labels**: good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

🎉 **#18343 introduces dynamic Expert Parallelism Load Balancing (EPLB)** for DeepSeek-V2/V3/R1 models.

As MoE (Mixture-of-Experts) models become more common, we’d love help extending EPLB support to other MoE models—such as Qwen3, Llama 4, and more.

This is a great **first good issue** for anyone interested in model internals or systems work. #18343 was built with generality in mind, so extending it to other models or quantization methods should be relatively straightforward.

---

### ✅ How to add support for a new model

Implement the `MixtureOfExperts` protocol. Specifically, you’ll need to:

- Expose relevant MoE configuration flags.
- Provide access to expert weights for EPLB to rearrange.
- Forward EPLB-related arguments into the `FusedMoE` layer.

📌 **Note on weight loading:**  
For models with **redundant experts**, you’ll need to carefully adjust the weight loading logic. `FusedMoE` returns an `expert_params_mapping` that reflects exp

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Image-Modality Throughput Benchmark

**Link**: https://github.com/vllm-project/vllm/issues/9778
**State**: closed
**Created**: 2024-10-28T23:39:19+00:00
**Closed**: 2024-11-05T19:30:04+00:00
**Comments**: 2
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

This is a subset of #8385. This issue is intended to track the effort of enabling throughput benchmark for image-modal models.

This is a reasonably large feature, and will span the work among multiple PRs.

### Alternatives

Ad-hoc scripts for each model.

### Additional context

see #8385 

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

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

## Issue #N/A: [Doc]: Does vllm CPU backend support Intel AMX?

**Link**: https://github.com/vllm-project/vllm/issues/14603
**State**: open
**Created**: 2025-03-11T08:34:12+00:00
**Comments**: 4
**Labels**: documentation

### Description

### 📚 The doc issue

I see pr #4971 has integrated some optimizations into vllm CPU backend, and I want to know whether vllm CPU supports intel AMX, and how to use it.
Thank you.

### Suggest a potential alternative/fix

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Installation]: Pytorch nightly version 2.6 meets error: error: can't copy '/tmp/tmpv5hlsgcm.build-lib/vllm/_core_C.abi3.so': doesn't exist or not a regular file

**Link**: https://github.com/vllm-project/vllm/issues/9180
**State**: closed
**Created**: 2024-10-09T05:23:31+00:00
**Closed**: 2024-11-11T01:48:06+00:00
**Comments**: 13
**Labels**: installation

### Description

### Your current environment

<details>

<summary>click here to view the env</summary>

```
Collecting environment information...
PyTorch version: 2.6.0.dev20241008+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: 16.0.1
CMake version: version 3.26.0
Libc version: glibc-2.31

Python version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:27:36) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.4.0-196-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.6.68
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: Tesla V100-SXM2-32GB
GPU 1: Tesla V100-SXM2-32GB
GPU 2: Tesla V100-SXM2-32GB
GPU 3: Tesla V100-SXM2-32GB

Nvidia driver version: 560.35.03
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.4.0
/usr/lib/x86_64-li

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: The usage of .transpose() and .view() consecutively is not recommended.

**Link**: https://github.com/vllm-project/vllm/issues/11978
**State**: closed
**Created**: 2025-01-13T01:41:52+00:00
**Closed**: 2025-01-13T06:24:12+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

<details>
<summary>Error information (Sorry, I cannot disclose more due to confidentiality reasons)</summary>

```text
Collecting environment information...
PyTorch version: 2.1.0
Is debug build: False
CUDA used to build PyTorch: Could not collect
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (aarch64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.2
Libc version: glibc-2.35

Python version: 3.10.12 (main, Sep 11 2024, 15:47:36) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-118-generic-aarch64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: 
GPU models and configuration: Could not collect
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

Versions of relevant libraries:
[pip3] min

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: quantization does not work with dummy weight format

**Link**: https://github.com/vllm-project/vllm/issues/9177
**State**: closed
**Created**: 2024-10-09T03:20:48+00:00
**Closed**: 2025-02-07T01:59:37+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### Model Input Dumps

_No response_

### 🐛 Describe the bug

found in ci, https://buildkite.com/vllm/ci-aws/builds/9653#01926d8d-2b87-4c7e-b5dd-bf56514236f0 , test_cpu_offload_gptq and test_cpu_offload_compressed_tensors do not work with dummy format.

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Multi-GPU error

**Link**: https://github.com/vllm-project/vllm/issues/612
**State**: closed
**Created**: 2023-07-28T09:48:12+00:00
**Closed**: 2023-08-07T22:44:40+00:00
**Comments**: 7
**Labels**: installation

### Description

While using tensor_parallel_size argument to load the vllm model, I was facing the issue in #557 stating something related to network address retrieval. According to [this comment](https://github.com/vllm-project/vllm/issues/570#issuecomment-1650973012) on #570 I trying building vllm from source and running it then.

The error goes away, but it does not load the model and gets stuck on that cell.

Is there any way I can get things working on multiple GPUs? I am able to run llama-2-7b, but in order to run llama-2-13b, I'll need to run it on multiple GPUs.

---

## Issue #N/A: [V1][Help Wanted] Porting missing sampling parameters to V1

**Link**: https://github.com/vllm-project/vllm/issues/13058
**State**: closed
**Created**: 2025-02-10T23:13:42+00:00
**Closed**: 2025-03-20T14:15:16+00:00
**Comments**: 10
**Labels**: good first issue, misc, v1

### Description

### Anything you want to discuss about vllm.

To switch the engine from V0 to V1, we need to comprehensively support the sampling parameters in https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py

While most of the key parameters are already supported, some of them are missing:

TODO (help wanted):
- [x] `n` (parallel sampling) #10980  @afeldman-nm 
- [x] `guided_decoding` (structured decoding) #12388  @aarnphm 
- [x] `logit_bias` #13079 @houseroad 
- [x] `min_p` #13191 @AoyuQC
- [ ] `bad_words` (originally implemented via logits processor) #13376 @22quinn 
- [x] `allowed_token_ids` (originally implemented via logits processor) #13210 @houseroad 

Parameters that will not be supported in V1:
* best_of
* logits_processors


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequentl

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): address already in use

**Link**: https://github.com/vllm-project/vllm/issues/8204
**State**: closed
**Created**: 2024-09-05T17:35:19+00:00
**Closed**: 2024-09-16T20:56:29+00:00
**Comments**: 8
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

This is a bug we encounter a lot in our ci, e.g. https://buildkite.com/vllm/ci-aws/builds/8098#0191bf43-446d-411d-80c7-3ba10bc392e8/192-1557

I have been tracking this for months, and try to add more logging information to help debugging.

from the logging information:


> [2024-09-05T00:38:34Z] INFO:     Started server process [60858]
> --
>   | [2024-09-05T00:38:34Z] INFO:     Waiting for application startup.
>   | [2024-09-05T00:38:34Z] INFO:     Application startup complete.
>   | [2024-09-05T00:38:34Z] ERROR:    [Errno 98] error while attempting to bind on address ('0.0.0.0', 44319): [errno 98] address already in use
>   | [2024-09-05T00:38:34Z] INFO:     Waiting for application shutdown.
>   | [2024-09-05T00:38:34Z] INFO:     Application shutdown complete.
>   | [2024-0

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Continue on Device agnostic API abstraction to current_platform.XXX

**Link**: https://github.com/vllm-project/vllm/issues/20708
**State**: open
**Created**: 2025-07-09T21:00:11+00:00
**Comments**: 8
**Labels**: RFC

### Description

co-author with @jikunshang 

### Motivation.

This RFC is aiming to reuse `GPUWorker` and `GPUModelRunner` for any GPGPU devices, such as CUDA, ROCM and Intel GPU(aka: XPU).
- By doing so, we can remove redundant duplication by adding a new XXXWorker/XXXModelRunner and derive from GPUWorker/GPUModelRunner
- Any feature implemented in GPUWorker/GPUModelRunner such as logitsProcessor, samplingOutput optimization, spec_decode can be shared to all GPGPU hardware.

**Status & Challenge**

- Previous RFC from Huawei has made significant work done through - https://github.com/vllm-project/vllm/issues/9268

- Currently, `GPUWorker` and `GPUModelRunner` is assumed that it will only be used by CUDA and RocM, so hard-coded to cuda API will be used in above two files. Ex:torch.cuda.XXX or tensor.to('cuda') 

### Proposed Change.

1. Add abstract API into platforms/interface.py and implement in cuda.py, rocm.py, xpu.py.
2. update any tensor.to('cuda') or tensor.cuda() to use tensor.to(current_platf

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][V1]: Kernel crashed when running  qwen2.5_vl model

**Link**: https://github.com/vllm-project/vllm/issues/14181
**State**: closed
**Created**: 2025-03-04T04:49:32+00:00
**Closed**: 2025-03-06T03:57:20+00:00
**Comments**: 4
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

I'm using the vllm 0.7.3 (enable_vllm_v1) to run qwen2_5_vl model. 

```shell
Initializing a V1 LLM engine (v0.7.3) with config: model='/qwen2_5-vl-72b', speculative_config=None, tokenizer='/qwen2_5-vl-72b', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=32000, download_dir=None, load_format=auto, tensor_parallel_size=4, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=fp8, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, serv

[... truncated for brevity ...]

---

## Issue #N/A: Can't launch OpenAI API server on newly installed vLLM in Docker - fastchat not found

**Link**: https://github.com/vllm-project/vllm/issues/537
**State**: closed
**Created**: 2023-07-20T21:35:25+00:00
**Closed**: 2023-08-02T18:05:31+00:00
**Comments**: 7

### Description

Hi

I have a Docker container that I created for vLLM. I built it a few days ago and it worked fine.  Today I rebuilt it to get the latest code changes, and now it's failing to launch the OpenAI server.  SSHing in to the docker and running the launch command directly shows the following error:

```
vllm@36b7089a5957:~/vllm (main ✔) ᐅ python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/vllm/vllm/vllm/entrypoints/openai/api_server.py", line 17, in <module>
    from fastchat.model.model_adapter import get_conversation_template
ModuleNotFoundError: No module named 'fastchat.model.model_adapter'
```

However I can launch the non-API server fine:
```
vllm@36b7089a5957:~/vllm (main ✔) ᐅ python -m vllm.en

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: `triton_scaled_mm` never used on ROCm

**Link**: https://github.com/vllm-project/vllm/issues/14397
**State**: open
**Created**: 2025-03-07T02:04:44+00:00
**Comments**: 5
**Labels**: bug, good first issue, unstale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 03-07 02:02:58 [__init__.py:207] Automatically detected platform rocm.
Collecting environment information...
PyTorch version: 2.5.1+rocm6.2
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.2.41133-dd7f95766

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 10.5.0-1ubuntu1~22.04) 10.5.0
Clang version: Could not collect
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-131-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI300X (gfx942:sramecc+:xnack-)
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: 6.2.41133
MIOpen runtime version: 3.2.0
Is XNNPACK availabl

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: [V1][Core] Structured decoding - Decouple Json from Json Object

**Link**: https://github.com/vllm-project/vllm/issues/13429
**State**: closed
**Created**: 2025-02-17T22:38:47+00:00
**Closed**: 2025-05-22T14:15:39+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

When using json_object = True for guided decoding (as per V0 guidelines), V1 (based on #12388 ) expects us to have a json_schema present for it as well. In nature both json_object and json need to be decoupled as per this [code snippet](https://github.com/vllm-project/vllm/pull/12388/files#diff-35f85e99eae8897d78a45f6a8d21bb69f9d8fe4a51e072bf299118dadac612f3R160) since json_object would inherently use the JsonGrammar compiled by xgrammar backend and would not require a Json Schema for it. 

```
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, guided_decoding=GuidedDecodingParams(
                           

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: JinaVLForRanking 400 BadRequest

**Link**: https://github.com/vllm-project/vllm/issues/20804
**State**: closed
**Created**: 2025-07-11T08:04:59+00:00
**Closed**: 2025-07-13T14:32:41+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

<details>

<summary>The output of <code>python collect_env.py</code></summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

This merger adds support for the JinaVL Reranker model ([[#20260](https://github.com/vllm-project/vllm/pull/20260)]). However, when I started the service and ran test calls, it returned a 400 Bad Request error.
### Online Serving
```text
vllm serve jinaai/jina-reranker-m0
```
### Request
```text
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/rerank' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "jinaai/jina-reranker-m0",
  "query": "slm markdown",
  "documents": {
    "content": [
      {
        "type": "image_url",
        "image_url": {
          "url": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
        }
      },
      {
        "type": "image_url",
        "image_url": {
        

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: stack trace for "Watchdog caught collective operation timeout"

**Link**: https://github.com/vllm-project/vllm/issues/12625
**State**: open
**Created**: 2025-01-31T18:23:56+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 01-31 10:13:08 __init__.py:183] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.12.8 (main, Dec  4 2024, 08:54:12) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-124-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3
GPU 4: NVIDIA H100 80GB HBM3
GPU 5: NVIDIA H100 80GB HBM3
GPU 6: NVIDIA H100 80GB HBM3
GPU 7: NVID

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Improve GPTQ implementation

**Link**: https://github.com/vllm-project/vllm/issues/15116
**State**: closed
**Created**: 2025-03-19T09:06:52+00:00
**Closed**: 2025-07-18T02:28:06+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

As we all know, quantizing some layers in the model will cause a large loss of accuracy. When using the AWQ algorithm, you can use the **modules_to_not_convert** attribute to avoid quantizing some layers. If the same function is also available in the GPTQ algorithm, I think it will be very convenient.

### Alternatives

I have pull a request [#12103](https://github.com/vllm-project/vllm/pull/12103)

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [RFC]: Implement disaggregated prefilling using Mooncake

**Link**: https://github.com/vllm-project/vllm/issues/10727
**State**: closed
**Created**: 2024-11-28T00:27:22+00:00
**Closed**: 2025-03-28T02:04:55+00:00
**Comments**: 2
**Labels**: RFC, stale

### Description

### Motivation.

Disaggregated prefilling/decoding is expected to achieve better performance (e.g., long documents) in LLM inference. [#5557](https://github.com/vllm-project/vllm/issues/5557) proposes a good paradigm. 

In addition, the Transfer Engine of [Mooncake](https://github.com/kvcache-ai/mooncake), which is a KVCache-centric disaggregated architecture for LLM serving, is open-sourced. 

Compared with NCCL, Mooncake Transfer Engine has the following features:
- a unified programming interface for data transfers between DRAM-to-DRAM (both local and remote), DRAM-to-GPU VRAM (both local and remote), and DRAM-to-remote NVMe devices
- support for TCP, RDMA, and NVMe-of protocols
- topology-aware path selection (link to our english doc, transfer_engine.md), aggregating bandwidth from multiple NICs

### Proposed Change.

The plan is to integrate vLLM with Mooncake. Initially we have implemented a prototype that replaces nccl with Transfer Engine in the data plane. In the future

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Benchmarks for audio models

**Link**: https://github.com/vllm-project/vllm/issues/16354
**State**: closed
**Created**: 2025-04-09T16:55:19+00:00
**Closed**: 2025-04-19T09:24:15+00:00
**Comments**: 2
**Labels**: help wanted, feature request, multi-modality

### Description

### 🚀 The feature, motivation and pitch

- Add audio datasets to `benchmarks/benchmark_dataset.py` to so we can run performance benchmarks on audio models as well.
- Add a benchmark similar to MMMU (#11196) but for audio models to evaluate their correctness.

### Alternatives

_No response_

### Additional context

cc @mgoin @ywang96 

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Roadmap] vLLM Roadmap Q2 2025

**Link**: https://github.com/vllm-project/vllm/issues/15735
**State**: closed
**Created**: 2025-03-29T00:21:57+00:00
**Closed**: 2025-07-01T21:08:45+00:00
**Comments**: 19

### Description

This page is accessible via [roadmap.vllm.ai](https://roadmap.vllm.ai/)

This is a living document! For each item here, we intend to link the RFC as well as discussion Slack channel in the [vLLM Slack](https://slack.vllm.ai)

---

#### Core Themes

**Path to vLLM v1.0.0**  
*We want to fully remove the V0 engine and clean up the codebase for unpopular and unsupported features. The v1.0.0 version of vLLM will be performant and easy to maintain, as well as modular and extensible, with backward compatibility.*

- [ ] V1 core feature set  
    - [x] Hybrid memory allocators  
    - [ ] ~Jump decoding~
    - [x] Redesigned native support for pipeline parallelism   
    - [x] Redesigned spec decode  
    - [ ] Redesigned sampler with modularity support
- [ ] Close the feature gaps and fully remove V0  
    - [x] Attention backends
    - [ ] Pooling models  
    - [ ] Mamba/Hybrid models  
    - [ ] (TBD) encoder and encoder decoder  
    - [x] Hardware support  
- [ ] Performance  
    - [ ]

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: A Graph Optimization System in vLLM using torch.compile

**Link**: https://github.com/vllm-project/vllm/issues/6378
**State**: closed
**Created**: 2024-07-12T15:33:58+00:00
**Closed**: 2025-04-13T02:17:27+00:00
**Comments**: 9
**Labels**: RFC, stale

### Description

### Motivation.

At a high level, we at Neural Magic are writing a custom compiler for Torch Dynamo to define a system within vLLM where we can write graph transformations. The main goal is a separation of concerns between high-level model definitions and certain performance-critical low-level decisions. This is especially important for optimizations that are particularly invasive to the model definitions, that break abstractions, that cross boundaries between layers, or that aren't universally valid or useful. If these optimizations are made as part of the model definitions, it becomes much more difficult to add new models.

We are working on the following for an initial set of optimizations using this system, described in detail in the Proposed Passes section.
* Fusing quantize operations onto LayerNorm kernels (both for fp8 and int8 and both static and dynamic quantization)
* Fusing the MLP section containing GEMM, SiLU, Mul, and quantize operations
* Rewriting Gemm + AllRedu

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: OOM Error with `Qwen/Qwen3-235B-A22B` on V1 Engine, Works on V0 Engine

**Link**: https://github.com/vllm-project/vllm/issues/18446
**State**: closed
**Created**: 2025-05-21T00:44:13+00:00
**Closed**: 2025-05-21T03:09:31+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
INFO 05-21 09:36:27 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 4.0.0
Libc version: glibc-2.35

Python version: 3.12.10 (main, Apr  9 2025, 08:55:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-119-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB
GPU 2: NVIDIA A100-SXM4-80GB
GPU 3: NVIDIA A100-SXM4-80GB
GPU 4: NVIDIA A100-SXM4-80GB
GPU 5: NVIDIA A100-SXM4-80GB
GPU 6: NVIDIA A100-SXM4-80GB
GPU 7: NVID

[... truncated for brevity ...]

---

