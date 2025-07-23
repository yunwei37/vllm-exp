# zero_comments - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 9
- Closed Issues: 21

### Label Distribution

- bug: 11 issues
- usage: 3 issues
- misc: 2 issues
- ci-failure: 1 issues
- installation: 1 issues
- feature request: 1 issues
- performance: 1 issues

---

## Issue #N/A: [Usage]: How can I input aip-key when use benchmark_serving.py to test my model

**Link**: https://github.com/vllm-project/vllm/issues/19530
**State**: open
**Created**: 2025-06-12T07:18:18+00:00
**Comments**: 0
**Labels**: usage

### Description

### Your current environment

```text
How can I input aip-key when use `benchmark_serving.py ` to test my model ?
```


### How would you like to use vllm

How can I input aip-key when use `benchmark_serving.py ` to test my model ?


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: ERNIE-4.5 does not run on an RTX Pro 6000 Blackwell

**Link**: https://github.com/vllm-project/vllm/issues/20712
**State**: open
**Created**: 2025-07-09T22:31:05+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
$ python collect_env.py
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 24.04.2 LTS (x86_64)
GCC version                  : (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version                : Could not collect
CMake version                : version 3.28.3
Libc version                 : glibc-2.39

==============================
       PyTorch Info
==============================
PyTorch version              : 2.7.0+cu128
Is debug build               : False
CUDA used to build PyTorch   : 12.8
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.3 (main, Jun 18 2025, 17:59:45) [GCC 13.3.0] (64-bit runtime)
Python platform              : Linux-6.8.0-62-generic-

[... truncated for brevity ...]

---

## Issue #N/A: [CI Failure]: Plugin Tests (2 GPUs) - models/test_oot_registration.py

**Link**: https://github.com/vllm-project/vllm/issues/20148
**State**: closed
**Created**: 2025-06-26T20:12:49+00:00
**Closed**: 2025-06-27T03:21:05+00:00
**Comments**: 0
**Labels**: ci-failure

### Description

### Name of failing test

`models/test_oot_registration.py::test_oot_registration_embedding`

### Basic information

- [ ] Flaky test
- [x] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

The `models/test_oot_registration.py::test_oot_registration_embedding` test seems to be failing in CI consistently with a context length OOM

https://buildkite.com/vllm/ci/builds/22737/steps/canvas?sid=0197acae-970a-43ee-9fef-108d8a58da0c#0197acae-98db-423d-8af9-eb4eb401f1b4/212-1320

```
[2025-06-26T16:27:15Z] ERROR 06-26 09:27:15 [core.py:519] ValueError: To serve at least one request with the models's max seq len (8192), (2.63 GiB KV cache is needed, which is larger than the available KV cache memory (1.64 GiB). Based on the available memory, the estimated maximum model length is 5088. Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
```

### 📝 History of failing test

Not sure

[... truncated for brevity ...]

---

## Issue #N/A: Error while trying to load phi3 model [Misc]: 

**Link**: https://github.com/vllm-project/vllm/issues/4366
**State**: closed
**Created**: 2024-04-25T13:31:41+00:00
**Closed**: 2024-04-25T13:32:18+00:00
**Comments**: 0
**Labels**: misc

### Description

### Anything you want to discuss about vllm.



Getting the model `Azma-AI/azma-phi-3-mini-3b-128k-250424` from hugging face
config.json: 100%|████████████████████████████████████| 3.43k/3.43k [00:00<00:00, 15.2MB/s]
configuration_phi3.py: 100%|██████████████████████████| 10.4k/10.4k [00:00<00:00, 37.6MB/s]
A new version of the following files was downloaded from https://huggingface.co/microsoft/Phi-3-mini-128k-instruct:
- configuration_phi3.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
Traceback (most recent call last):
  File "/root/.cache/pypoetry/virtualenvs/azma-xS3fZVNL-py3.10/bin/uvicorn", line 8, in <module>
    sys.exit(main())
  File "/root/.cache/pypoetry/virtualenvs/azma-xS3fZVNL-py3.10/lib/python3.10/site-packages/click/core.py", line 1157, in __call__
    return self.main(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/azma-xS3fZVNL-py3.10/lib

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: Local environment installation succeeded, Existing docker environment failed log

**Link**: https://github.com/vllm-project/vllm/issues/13574
**State**: closed
**Created**: 2025-02-20T00:46:13+00:00
**Closed**: 2025-02-20T00:46:42+00:00
**Comments**: 0
**Labels**: installation

### Description

### Your current environment

**Local environment:**
- System: Ubuntu 22.04  
- CUDA: 12.4  
- PyTorch: 2.4.0  
- CMake: 2.31.4  
- GCC: 11.4.0  


**Docker environment:**  
- System: Ubuntu 24.04  
- CUDA: 12.6  
- PyTorch: 2.6.0  
- CMake: 2.31.4  
- GCC: 13.3.0 

### How you are installing vllm

Code version: 2025.2.19, main branch  
Local environment:  
- System: Ubuntu 22.04  
- CUDA: 12.4  
- PyTorch: 2.4.0  
- CMake: 2.31.4  
- GCC: 11.4.0  
Compiled and installed successfully using the documentation.

However, errors occurred when attempting to install in the Docker environment:  
**Docker environment:**  
- System: Ubuntu 24.04  
- CUDA: 12.6  
- PyTorch: 2.6.0  
- CMake: 2.31.4  
- GCC: 13.3.0  

### Issues from the Docker environment:
1. Issue: Configured the `nvcc` environment variable, but the folder `/usr/local/cuda-12.6/bin/nvcc` was not found.  
   Solution: 
   Edit `~/.bash_profile` or `~/.bashrc`  
   Modify the configuration to:  
   ```bash
   export PATH=/usr/loca

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Engine timeout error due to request step residual

**Link**: https://github.com/vllm-project/vllm/issues/6254
**State**: closed
**Created**: 2024-07-09T09:30:12+00:00
**Closed**: 2024-07-11T13:46:32+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 3.29.2
Libc version: glibc-2.31

Python version: 3.9.2 (default, Feb 28 2021, 17:03:44)  [GCC 10.2.1 20210110] (64-bit runtime)
Python platform: Linux-5.4.143.bsk.8-amd64-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 10.0.130
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L40
GPU 1: NVIDIA L40
GPU 2: NVIDIA L40
GPU 3: NVIDIA L40

Nvidia driver version: Could not collect
cuDNN version: /usr/lib/x86_64-linux-gnu/libcudnn.so.7.6.0
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU op-mode(s):

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: MiniCPM-o int4

**Link**: https://github.com/vllm-project/vllm/issues/17358
**State**: closed
**Created**: 2025-04-29T08:00:43+00:00
**Closed**: 2025-04-29T18:21:44+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

jetson orin 64G
vllm 0.8.5


### 🐛 Describe the bug

When i load MiniCPM-o int4 using "dtype": "float16", and "quantization": "gptq". I encounted 


```
File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/model_loader/loader.py", line 455, in load_model
    loaded_weights = model.load_weights(
  File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/minicpmo.py", line 531, in load_weights
    return loader.load_weights(weights)
  File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/utils.py", line 261, in load_weights
    autoloaded_weights = set(self._load_module("", self.module, weights))
  File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/utils.py", line 222, in _load_module
    yield from self._load_module(prefix,
  File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/utils.py", line 222, in _load_module
    yield from self._load_module(prefix,
  File "/usr/l

[... truncated for brevity ...]

---

## Issue #N/A: The same model cannot be loaded by two different users

**Link**: https://github.com/vllm-project/vllm/issues/2232
**State**: closed
**Created**: 2023-12-21T09:40:34+00:00
**Closed**: 2024-03-23T18:43:12+00:00
**Comments**: 0

### Description

As pointed out here, the way lockfiles are created prevents the second user from loading any models that a previous user has loaded at any point: https://github.com/vllm-project/vllm/issues/2179

This is still an issue with the only workaround being to force-delete the lockfile created by another user.

---

## Issue #N/A: [ERROR] [ -4263953,Not implemented in the current version ] nvmlDeviceGetHandleByPciBusId() This is unlikely to affect the main functionalities of user applications.

**Link**: https://github.com/vllm-project/vllm/issues/2325
**State**: closed
**Created**: 2024-01-03T03:51:15+00:00
**Closed**: 2024-03-28T13:39:31+00:00
**Comments**: 0

### Description

when i run this command:
```
llm = LLM(model="qwen/Qwen-7B-Chat", revision="v1.1.8", trust_remote_code=True,quantization='awq')
```

the error below:
```
WARNING 01-03 11:47:54 config.py:171] awq quantization is not fully optimized yet. The speed can be slower than non-quantized models.
INFO 01-03 11:47:54 llm_engine.py:73] Initializing an LLM engine with config: model='/data/share/rwq/Qwen-7B-Chat', tokenizer='/data/share/rwq/Qwen-7B-Chat', tokenizer_mode=auto, revision=v1.1.8, tokenizer_revision=None, trust_remote_code=True, dtype=torch.float16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=1, quantization=awq, seed=0)
WARNING 01-03 11:47:54 tokenizer.py:79] Using a slow tokenizer. This might cause a significant slowdown. Consider using a fast tokenizer instead.
2024-01-03 11:48:06 [ERROR] [ -4263953,Not implemented in the current version ] nvmlDeviceGetHandleByPciBusId() This is unlikely to affect the main functionalities of user applications.

[... truncated for brevity ...]

---

## Issue #N/A: 1

**Link**: https://github.com/vllm-project/vllm/issues/2575
**State**: closed
**Created**: 2024-01-24T08:25:18+00:00
**Closed**: 2024-01-24T08:25:33+00:00
**Comments**: 0

### Description

No description provided.

---

## Issue #N/A: Setting up CodeGen for vllm but getting error "KeyError: 'transformer.h.0.attn.causal_mask'"

**Link**: https://github.com/vllm-project/vllm/issues/1981
**State**: closed
**Created**: 2023-12-08T05:07:27+00:00
**Closed**: 2024-03-25T10:26:43+00:00
**Comments**: 0

### Description

I have created a new 'codegen' adaptor for vllm but am getting the error "KeyError: 'transformer.h.0.attn.causal_mask'" when trying to do my first test. Need help to find and fix the issue.

Exception:
INFO 12-08 05:00:54 llm_engine.py:73] Initializing an LLM engine with config: model='Salesforce/codegen-350M-mono', tokenizer='Salesforce/codegen-350M-mono', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=auto, tensor_parallel_size=1, quantization=None, seed=0)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Traceback (most recent call last):
  File "/content/drive/MyDrive/Workspace/vllm-main/./examples/offline_inference.py", line 14, in <module>
    llm = LLM(model="Salesforce/codegen-350M-mono")
  File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/llm.py", line 93, in __init__
    self.llm

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: `samplers/test_logprobs.py` fail on H100

**Link**: https://github.com/vllm-project/vllm/issues/6408
**State**: closed
**Created**: 2024-07-13T04:13:43+00:00
**Closed**: 2024-07-15T17:14:50+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.5
Libc version: glibc-2.35

Python version: 3.11.9 | packaged by conda-forge | (main, Apr 19 2024, 18:36:13) [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-5.15.0-107-generic-x86_64-with-glibc2.35
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
GPU 7: NVIDIA H100 80GB HBM3

Nvidia driver version: 550.54.15
cuDNN version: Probably one of the fo

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: fp8 w8a8 quantized Qwen2.5-VL hits AssertionError

**Link**: https://github.com/vllm-project/vllm/issues/17595
**State**: open
**Created**: 2025-05-02T16:54:40+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

vLLM v0.8.4 on Tesla L40S.

### 🐛 Describe the bug

See also these closed issues: #7550 #15264

```
ERROR 05-02 09:48:39 [engine.py:160] AssertionError()
ERROR 05-02 09:48:39 [engine.py:160] Traceback (most recent call last):
ERROR 05-02 09:48:39 [engine.py:160]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 158, in start
ERROR 05-02 09:48:39 [engine.py:160]     self.run_engine_loop()
ERROR 05-02 09:48:39 [engine.py:160]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 221, in run_engine_loop
ERROR 05-02 09:48:39 [engine.py:160]     request_outputs = self.engine_step()
ERROR 05-02 09:48:39 [engine.py:160]                       ^^^^^^^^^^^^^^^^^^
ERROR 05-02 09:48:39 [engine.py:160]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 247, in engine_step
ERROR 05-02 09:48:39 [engine.py:160]     raise e
ERROR 05-02 09:48:39 [engin

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: 使用qwen2.5-omni对音频识别，cpu会被打满。

**Link**: https://github.com/vllm-project/vllm/issues/19552
**State**: open
**Created**: 2025-06-12T12:40:34+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
vllm serve Qwen2.5-Omni-3B --dtype bfloat16 --gpu_memory_utilization=0.9 --tensor-parallel-size 1 --swap-space 0

A800机器，80G显存，cpu200核。
```

</details>


### 🐛 Describe the bug

调用推理服务，发现cpu利用率达到99%。会支持将audio的处理放在gpu里吗？

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Question][Schedule] vLLM cannot schedule a whole sequence group but can schedule some of its sequences

**Link**: https://github.com/vllm-project/vllm/issues/1423
**State**: closed
**Created**: 2023-10-20T03:00:13+00:00
**Closed**: 2024-03-13T11:40:25+00:00
**Comments**: 0

### Description

As the [paper](https://arxiv.org/pdf/2309.06180.pdf) wrote,

> The sequences within one sequence group are always preempted or rescheduled together due to potential memory sharing across those sequences.

In such a condition:

* **Total** GPU memory is not enough to hold a sequence group
* **Total** GPU memory is enough to hold some of sequences in that group

This may happen in sequence groups with a large size. According to the paper, this sequence group would not be scheduled, and the scheduler would be **stuck**.

Would the following method be better for availability while still utilizing the advantage of memory sharing:

* First, try scheduling a sequence group.
* If all sequence groups to be scheduled can not fit in GPU memory, select one sequence group and schedule as many sequences of this group.

---

## Issue #N/A: [Misc]: RoPE vs Sliding Windows

**Link**: https://github.com/vllm-project/vllm/issues/12328
**State**: closed
**Created**: 2025-01-22T19:44:02+00:00
**Closed**: 2025-02-17T16:33:46+00:00
**Comments**: 0
**Labels**: misc

### Description

Hi,

As context lengths increase, it looks like different models are going about it in different ways. For example, Qwen uses a sliding window in their config.json file while Llama uses RoPE.

I was curious how they work in contrast with each other, if they can be combined, what types of RoPE scaling exist, and how all of these parameters can be optimized separately or in conjunction with each other.

I am also curious how setting the `--rope-scaling` and `--rope-theta` interacts with the configs if it is already set or using sliding windows. I can't find too much information regarding the combination of all of these settings so any help would be awesome.



```json
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 27648,
  "max_position_embeddings": 32768,
  "max_window_layers": 70,
  "model_type": "qwen2",


[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: make `test_openai_schema.py` pass, enable it in CI

**Link**: https://github.com/vllm-project/vllm/issues/18162
**State**: closed
**Created**: 2025-05-14T18:04:12+00:00
**Closed**: 2025-05-22T18:34:07+00:00
**Comments**: 0
**Labels**: bug

### Description

This is a follow up to PR #17664 and issues like #17037 and #17038.

As of the time of this writing, the only sub-test in `tests/entrypoints/openai/test_openai_schema.py` that consistently fails is `POST /tokenize` caused by https://github.com/vllm-project/vllm/blob/98ea35601cdb34fdd618f965e7bcc3cb02a677fc/vllm/entrypoints/chat_utils.py#L1083 when `part_type` is `"file"`. The expected response is 200 as documented by the OpenAPI spec.

How should we make this test pass? This is a requirement to eventually enabling the test in CI by removing `--ignore=entrypoints/openai/test_openai_schema.py` in `test-pipeline.yaml`.

<details>

```
============================================================================================ FAILURES =============================================================================================
_____________________________________________________________________ test_openapi_stateless (verbose_name='POST /tokenize') _______________________________________

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: minicpmv2.6 BNB in-flight quantization error

**Link**: https://github.com/vllm-project/vllm/issues/9914
**State**: closed
**Created**: 2024-11-01T11:07:42+00:00
**Closed**: 2024-11-04T03:36:42+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

### Model Input Dumps

_No response_

### 🐛 Describe the bug

After merging https://github.com/vllm-project/vllm/pull/9891 , I tried the in-flight quantization with minicpmv and encountered the following error:

```shell
[rank0]:   File "/vllm/vllm/model_executor/model_loader/loader.py", line 1105, in _load_weights
[rank0]:     model.load_weights(qweight_iterator)
[rank0]:   File "/vllm/vllm/model_executor/models/minicpmv.py", line 634, in load_weights
[rank0]:     param = params_dict[name]
[rank0]: KeyError: 'vpm.encoder.layers.0.mlp.fc1.weight'
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
```



## Reproduce code

```python 
MODEL_NAME = "openbmb/MiniCPM-V-2_6"
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
)
```

 It seems `mllama` has the same issue.

[... truncated for brevity ...]

---

## Issue #N/A: Seeing similar latency for output len=1 and output len=128. Is something wrong with benchmark latency script?

**Link**: https://github.com/vllm-project/vllm/issues/1805
**State**: closed
**Created**: 2023-11-28T01:44:42+00:00
**Closed**: 2024-03-25T09:45:40+00:00
**Comments**: 0

### Description

Hi I'm trying to benchmark Llama 7B's latency using [benchmark_latency.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_latency.py) script but I'm seeing similar latency for output len=1 and output len=128 which makes no sense


Within the script I changed the `model_max_len=2000` because that's the input len I want to benchmark with. Here's the commands I used with vllm 0.2.0 on g5.12xlarge (4 A10G GPUs) on AWS

```
python3 benchmark_latency.py --model "meta-llama/Llama-2-7b-hf" --input-len 2000 --output-len 128 -tp 4 --num-iters 10 --batch-size 1
```
gives 462ms

and 
```
python3 benchmark_latency.py --model "meta-llama/Llama-2-7b-hf" --input-len 2000 --output-len 1 -tp 4 --num-iters 10 --batch-size 1
```
gives 459 ms

I would expect output len=1 latency to be much lower than that of output len=128. Can someone please help me understand this?

---

## Issue #N/A: [Bug]: InternVL2-2B outputs gibberish with tensor parallel inference

**Link**: https://github.com/vllm-project/vllm/issues/8017
**State**: closed
**Created**: 2024-08-30T03:32:21+00:00
**Closed**: 2024-09-02T15:48:57+00:00
**Comments**: 0
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

**Reproduce**
- Just run `examples/offline_inference_vision_language.py` with `tensor_parallel_size=2`.
- The inference with `tensor_parallel_size=1` works normally.

**Outputs**
```
Processed prompts: 100%|████████████████████████████████████████| 1/1 [00:01<00:00,  1.52s/it, est. speed input: 1192.65 toks/s, output: 26.96 toks/s]
1.
1.
1/2
3/2
1
  for example
�2/定有了一个iSA诉快的队/纳厄/否化，4.
INFO 08-30 03:22:42 multiproc_worker_utils.py:136] Terminating local vLLM worker processes
(VllmWorkerProcess pid=9476) INFO 08-30 03:22:42 multiproc_worker_utils.py:237] Worker exiting
```

**The root issue**
- This is broken by the `split_qkv` function for `internlm2` backbone introduced in #7187 to make compatible with awq model.

### Before submitting a new issue...

- [X] Ma

[... truncated for brevity ...]

---

## Issue #N/A: BUG python -m vllm.entrypoints.openai.api_server --model /workspace/api/models/Qwen/Qwen-7B-Chat/ --trust-remote-code  vllm==0.2.2 torch2.1.0+cuda118

**Link**: https://github.com/vllm-project/vllm/issues/1738
**State**: closed
**Created**: 2023-11-21T10:03:52+00:00
**Closed**: 2023-11-23T01:00:33+00:00
**Comments**: 0

### Description

Traceback (most recent call last):
  File "/usr/lib/python3.8/runpy.py", line 185, in _run_module_as_main
    mod_name, mod_spec, code = _get_module_details(mod_name, _Error)
  File "/usr/lib/python3.8/runpy.py", line 111, in _get_module_details
    __import__(pkg_name)
  File "/usr/local/lib/python3.8/dist-packages/vllm/__init__.py", line 3, in <module>
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
  File "/usr/local/lib/python3.8/dist-packages/vllm/engine/arg_utils.py", line 6, in <module>
    from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
  File "/usr/local/lib/python3.8/dist-packages/vllm/config.py", line 9, in <module>
    from vllm.utils import get_cpu_memory
  File "/usr/local/lib/python3.8/dist-packages/vllm/utils.py", line 8, in <module>
    from vllm import cuda_utils
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory






Traceback (most recent call last):
  File "/usr/lib/pyth

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]:

**Link**: https://github.com/vllm-project/vllm/issues/19272
**State**: closed
**Created**: 2025-06-06T11:13:15+00:00
**Closed**: 2025-06-06T12:47:25+00:00
**Comments**: 0
**Labels**: usage

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.
i use 'vllm sever' run Qwen3


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: missing latest tag from cpu docker registry

**Link**: https://github.com/vllm-project/vllm/issues/19869
**State**: closed
**Created**: 2025-06-19T14:40:17+00:00
**Closed**: 2025-06-23T21:15:38+00:00
**Comments**: 0
**Labels**: usage

### Description

### Your current environment

‘latest’ tag is missing from https://gallery.ecr.aws/q9t5s3a7/vllm-cpu-release-repo,  which requires having to keep changing the image manually on every release.

we expect the CI/CD not only to create a versioned tag, but also to tag latest once a new release is out.

### How would you like to use vllm

I want to run inference


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: vllm torch nightly package not in sync issues

**Link**: https://github.com/vllm-project/vllm/issues/18772
**State**: open
**Created**: 2025-05-27T17:30:17+00:00
**Comments**: 0
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

When a new pip package is included, and if it's imported by the tests that runs in both regular vllm ci and torch-nightly, the torch-nightly failed due to no module found.

Currently we have seperate txt file torch_nightly_test.txt to track the dependency, and vllm ci uses test.txt which is generated by test.in. The reason we set it in this way bc some dependency there has strict dependency on pytorch stable, and install them override the pytorch nightly setup.


## Proposal
One proposal is set up a  test_isolate_requirement.in, and move all non-pytorch-stable-dependent packages into the test_isolate_requirement.in, and lket test.in depends on that

when contributor wants to add a dependency -> if its not depends on pytorch stable, added it to test_isolate_requirement.in, generate both torch_nightly_test.txt and test.txt



### Alternatives

keep track of the dependency

### Additional context

n

### Before submitting a new issue...

- [x] Make

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: Severe performance drop on 1x A100 80GB with Qwen3-14B-AWQ at >1 concurrency (v0.9.1)

**Link**: https://github.com/vllm-project/vllm/issues/20469
**State**: open
**Created**: 2025-07-04T05:53:42+00:00
**Comments**: 0
**Labels**: performance

### Description

### Report of performance regression

![Image](https://github.com/user-attachments/assets/7dce9a2b-327d-49c2-acfe-9bc12cb2f6a7)

We observed a significant drop in output tokens per second when serving Qwen/Qwen3-14B-AWQ on a single A100 80GB GPU using vLLM v0.9.1 with --max-model-len 16384.

At concurrency=1, the model achieves ~52 output tokens per second. However, this drops sharply to ~12 at concurrency=5 and ~3 at concurrency=25. This performance is comparable to or worse than a 2x A30 setup, and significantly below the 2x A100 80GB (TP=2) configuration, which maintains stable output tokens per second (~38) across all concurrency levels.

In addition, Time-To-First-Token (TTFT) is already high at concurrency=1 (~3345 ms) and increases substantially with concurrency, reaching over 34 seconds at concurrency=25. In contrast, the 2x A100 setup maintains TTFT around ~100 ms across all levels.

vLLM reports a supported max concurrency of 26 for this configuration, so we expected it to ha

[... truncated for brevity ...]

---

## Issue #N/A: Any plan to support paged attention for prefill?

**Link**: https://github.com/vllm-project/vllm/issues/1598
**State**: closed
**Created**: 2023-11-09T03:48:47+00:00
**Closed**: 2023-11-13T03:20:32+00:00
**Comments**: 0

### Description

First of all, thank you for the great work and the recent integration for the paged-flash attention v2 kernel!

I am wondering if there is any plan to support paged attention for prefill, which can compute multiple tokens in each batch in parallel (like flash_attn_with_kvcache did). I did a quick check over the codebase and found it seems that paged_attention_v2_kernel expects one token for each request.

In some cases like speculative decoding and chunked prefill, it would be ideal to compute multiple tokens in each request in parallel. 

---

## Issue #N/A: [Bug]: V1 on AMD MI300A complains that cupy is not present

**Link**: https://github.com/vllm-project/vllm/issues/17875
**State**: open
**Created**: 2025-05-09T03:05:36+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
INFO 05-08 23:02:38 [__init__.py:239] Automatically detected platform rocm.
Collecting environment information...
PyTorch version: 2.7.0a0+git295f2ed
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.4.43482-0f2d60242

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.0 25133 c7fe45cf4b819c5991fe208aaa96edf142730f1d)
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.12.10 (main, Apr  9 2025, 08:55:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-4.18.0-553.47.1.1toss.t4.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI300A (gfx942:sramecc+:xnack-)
Nvidia driver version: C

[... truncated for brevity ...]

---

## Issue #N/A: Profile memory usage

**Link**: https://github.com/vllm-project/vllm/issues/59
**State**: closed
**Created**: 2023-05-03T06:00:44+00:00
**Closed**: 2023-05-19T17:35:45+00:00
**Comments**: 0

### Description

No description provided.

---

## Issue #N/A: [New Model]: OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B

**Link**: https://github.com/vllm-project/vllm/issues/18201
**State**: open
**Created**: 2025-05-15T12:27:21+00:00
**Comments**: 0

### Description

### The model to consider.

Hey, I want to make a request for adding a new model [OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B). Thank you!

### The closest model vllm already supports.

The closest model vllm already supports would be https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava_onevision.py. 

### What's your difficulty of supporting the model you want?

New processor. 1B InternVL vision processor. 
Also 16 tokens for a frame that's something I really want to test using vllm. 

from official Huggingface model card: "_VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B is constructed upon InternVideo2-1B and Qwen2.5-7B, employing only **16 tokens per frame**. By leveraging Yarn to extend the context window to 128k (Qwen2's native context window is 32k), our model supports input sequences of up to approximately **10,000 frames.**_"

Thank you!

### Before submitting a new issu

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Enable cutom_op of rotary_embedding goes error for Qwen3-4B

**Link**: https://github.com/vllm-project/vllm/issues/21101
**State**: open
**Created**: 2025-07-17T07:45:23+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

torch   version  2.7.1
vllm version 0.9.2rc2.dev304+g28a6d5423.cu124 compile myself depends on newset mast.

### 🐛 Describe the bug

I modify vllm/config.py to enable rotary_embedding op to enable custom_op.
modify:
--- a/vllm/config.py
+++ b/vllm/config.py
@@ -4624,6 +4624,7 @@ class VllmConfig:
             not self.model_config.enforce_eager:
             # By default, V1 uses piecewise CUDA graphs. If full_cuda_graph
             # is set to True, full CUDA graphs will be used.
+            self.compilation_config.custom_ops=["none","+rotary_embedding”]

run for Qwen3-4B for v1 mode torch.compile. return error as follows:
ERROR 07-17 15:36:30 [core.py:592]   File "/root/.virtualenvs/torch-env/lib/python3.10/site-packages/torch/fx/graph.py", line 1172, in erase_node
ERROR 07-17 15:36:30 [core.py:592]     raise RuntimeError(
ERROR 07-17 15:36:30 [core.py:592] torch._inductor.exc.InductorError: RuntimeError: Tried to erase Node getitem_4 but it still had 

[... truncated for brevity ...]

---

