# unstale - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 20
- Closed Issues: 10

### Label Distribution

- unstale: 30 issues
- bug: 11 issues
- usage: 6 issues
- feature request: 4 issues
- performance: 2 issues
- tool-calling: 2 issues
- RFC: 2 issues
- structured-output: 2 issues
- ray: 1 issues
- installation: 1 issues

---

## Issue #N/A: [Performance]: phi 3.5 vision model consuming high CPU RAM and the process getting killed

**Link**: https://github.com/vllm-project/vllm/issues/9190
**State**: open
**Created**: 2024-10-09T12:19:09+00:00
**Comments**: 37
**Labels**: performance, unstale

### Description

### Proposal to improve performance

I am trying to run phi3.5 vision instruct model with around 10k prompts. What I noticed with the increase in prompts my CPU RAM consumption keeps increasing and eventually the process gets killed. Its running fine for say small sample like 1000 prompts. My system configuration is 48 GB VRAM and 64GB CPU RAM. Noticed a similar pattern with PIXTRAL-12B-2409. Has anyone faced this issue?

I have tried the implementation by passing in batches of 1000 to llm.generate but still the CPU RAM keeps increasing
Below is the code implementation:
Ima using two images per prompt
from vllm import LLM, SamplingParams
llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 4},
    )
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
outputs = llm.generate(prompt_list, sampling_params=sampling_params)

[... truncated for brevity ...]

---

## Issue #N/A: HQQ quantization support

**Link**: https://github.com/vllm-project/vllm/issues/2871
**State**: closed
**Created**: 2024-02-14T15:50:49+00:00
**Closed**: 2025-01-14T13:32:35+00:00
**Comments**: 8
**Labels**: unstale

### Description

As we have a few models with Half-Quadratic Quantization (HQQ) out there, VLLM should also support them:

```sh
api_server.py: error: argument --quantization/-q: invalid choice: 'hqq' (choose from 'awq', 'gptq', 'squeezellm', None)
```

E.g.
* https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-2bit-HQQ

---

## Issue #N/A: [Usage]:  why speculate decoding is slower than normal decoding？

**Link**: https://github.com/vllm-project/vllm/issues/8439
**State**: open
**Created**: 2024-09-13T03:43:26+00:00
**Comments**: 14
**Labels**: usage, unstale

### Description

### Your current environment

The startup command is as follows: it initiates both a standard 7B model and an n-gram speculate model. Speed tests  discover that the speculate model performs more slowly."
```text
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 9000 --model Qwen2-7B-Instruct -tp 1 --gpu_memory_utilization 0.9

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 9002 --model Qwen2-7B-Instruct -tp 1 --speculative_model [gram] --use-v2-block-manager --num_speculative_tokens 5 --ngram-prompt-lookup-max 4 --gpu_memory_utilization 0.9

result
7b:
first token:  0.04074668884277344s
decode time:  14.328832149505615s
output token:  1000
decode speed:  69.78935823702163 token/s

spec 7b
first token:  0.02350592613220215s
decode time:  15.324904918670654s
output token:  947
decode speed:  61.794836902788866 token/s
```


### How would you like to use vllm

I want to run inference of a

[... truncated for brevity ...]

---

## Issue #N/A: Loading models from an S3 location instead of local path

**Link**: https://github.com/vllm-project/vllm/issues/3090
**State**: open
**Created**: 2024-02-28T18:20:13+00:00
**Comments**: 15
**Labels**: unstale

### Description

### Discussed in https://github.com/vllm-project/vllm/discussions/3072

<div type='discussions-op-text'>

<sup>Originally posted by **petrosbaltzis** February 28, 2024</sup>
Hello,

The VLLM library gives the ability to load the model and the tokenizer either from a local folder or directly from HuggingFace.
```
["python", "-m", "vllm.entrypoints.openai.api_server", \
"--host=0.0.0.0", \
"--port=8080", \
"--model=<local_path>", \
"--tokenizer=<local_path>",
]
```

I wonder if this functionality can be extended to support s3 locations so that when we initialize the API server, we pass the proper S3 location.

```
["python", "-m", "vllm.entrypoints.openai.api_server", \
"--host=0.0.0.0", \
"--port=8080", \
"--model=<s3://bucket/prefix>", \
"--tokenizer=<s3://bucket/prefix>",
]
```

Petros</div>

---

## Issue #N/A: [Usage]: how to use tool calling with auto option, setting the tool works

**Link**: https://github.com/vllm-project/vllm/issues/12349
**State**: open
**Created**: 2025-01-23T08:37:03+00:00
**Comments**: 1
**Labels**: usage, unstale, tool-calling

### Description

### Your current environment

-


### How would you like to use vllm

I am trying to use tool calling to test a qwen model. It works when specified the tool but normal queries don’t work. How to use auto mode? 

If it’s not supported when we can expect this? 

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [RFC]: Add runtime weight update API

**Link**: https://github.com/vllm-project/vllm/issues/5723
**State**: closed
**Created**: 2024-06-20T20:22:40+00:00
**Closed**: 2025-01-17T02:00:23+00:00
**Comments**: 23
**Labels**: RFC, unstale

### Description

### Motivation.

In online RL training, vLLM can significantly accelerate the rollout stage. To achieve this, we need weight sync from main training process to vLLM worker process, and then call the existing API in vLLM to update the weights by
`model_runner.model.load_weights `
An example of such implementation can be found in OpenRLHF, [https://github.com/OpenLLMAI/OpenRLHF/blob/main/openrlhf/trainer/ray/vllm_worker_wrap.py](vllm_worker_wrap)

However, user has to monkey patch vLLM worker to introduce such behavior. It would be great if vLLM naturally supports weight sync at runtime.

### Proposed Change.

1. Add a NCCL-based weight sync process group during vLLM initialization, so that main process can dist.broadcast weight to vLLM worker process later
2. Expose a weight sync API, for example:
`def update_weight(self, name, dtype, shape)`

then in master process, user can achieve weight sync via the following (modified from OpenRLHF):
```
for name, param in model.named_par

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: PaliGemma2 not working with OpenAI Docker serve

**Link**: https://github.com/vllm-project/vllm/issues/12052
**State**: open
**Created**: 2025-01-14T19:22:48+00:00
**Comments**: 19
**Labels**: bug, unstale

### Description

### Your current environment

Just using Docker image 0.6.6post1


### Model Input Dumps

_No response_

### 🐛 Describe the bug

Just try to run https://huggingface.co/google/paligemma2-3b-pt-896 using Docker vllm image. My docker compose follows:

```
services:
  app:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./cache:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=hf_
    ipc: host
    command:
      - --host
      - 0.0.0.0
      - --model
      - google/paligemma2-3b-pt-896
      - --limit-mm-per-prompt
      - 'image=1'
      - --trust-remote-code
      - --max-model-len
      - "8192"

```

It does NOT work, the issue is the same reported in the pull request here: https://github.com/vllm-project/vllm/pull/11142#issuecomment-2541342321
and is:
`ValueError: As of transformers v4.44, default chat template is no longer allowed, so you must provide a chat template if th

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Llama3.3 Tool calling support or a Geneneric and extensible llama tool calling support

**Link**: https://github.com/vllm-project/vllm/issues/11799
**State**: open
**Created**: 2025-01-07T07:01:45+00:00
**Comments**: 1
**Labels**: feature request, unstale, tool-calling

### Description

### 🚀 The feature, motivation and pitch

We have customer moving from llama3.1/3.2 to 3.3 and further when available

### Alternatives

Not yet explored

### Additional context

A generic way where we can use use tool calling support against llms instead of using specific params like 
--tool-call-parser llama3_json  /instead of --tool-call-parser <whatever model supports> as an external reference via chat template or so ?

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: who to run cluster withou docker

**Link**: https://github.com/vllm-project/vllm/issues/12053
**State**: open
**Created**: 2025-01-14T19:24:08+00:00
**Comments**: 2
**Labels**: usage, unstale

### Description

### Your current environment

if i cannot use docker ,how to run vllm on multiple nodes ?



### How would you like to use vllm

if i cannot use docker ,how to run vllm on multiple nodes ?



### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: Guided decoding is broken because tokenizers can't be pickled

**Link**: https://github.com/vllm-project/vllm/issues/7557
**State**: open
**Created**: 2024-08-15T14:16:17+00:00
**Comments**: 3
**Labels**: bug, structured-output, unstale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.3.1+cpu
Is debug build: False
CUDA used to build PyTorch: Could not collect
ROCM used to build PyTorch: N/A

OS: Fedora release 39 (Thirty Nine) (x86_64)
GCC version: (GCC) 13.3.1 20240522 (Red Hat 13.3.1-1)
Clang version: 17.0.6 (Fedora 17.0.6-2.fc39)
CMake version: version 3.29.6
Libc version: glibc-2.38

Python version: 3.11.8 (main, Mar 27 2024, 15:03:48) [GCC 13.2.1 20231205 (Red Hat 13.2.1-6)] (64-bit runtime)
Python platform: Linux-6.7.11-200.fc39.x86_64-x86_64-with-glibc2.38
Is CUDA available: False
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: GPU 0: NVIDIA GeForce MX330
Nvidia driver version: 545.23.08
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: Mode/flag/option to maximize throughput while allowing large latency?

**Link**: https://github.com/vllm-project/vllm/issues/6945
**State**: closed
**Created**: 2024-07-30T12:11:17+00:00
**Closed**: 2025-01-14T14:28:05+00:00
**Comments**: 4
**Labels**: performance, unstale

### Description

### Proposal to improve performance

Hi thank you for the great project! I would like to use vllm to run inference to test models on datasets. For example, say evaluating whether a prompt is good or not on the GSM8K dataset. I currently start a vllm openai-compatible server, and let python code to communicate with it.

Therefore, I do not care about latency, but only care about throughput. It seems that vllm's openai-compatible server cares about latency, thus I wonder that makes throughput suboptimal?

I know there is also a `LLM` class for batch inference. However, I hope to make vllm isolated from my main python environment (since it requires strict cuda/pytorch/etc), thus put it in a separate docker container via the official vllm openai docker image. So another related question is that, will the LLM class be different from using the vllm server and feed in all requests quickly?

### Report of performance regression

_No response_

### Misc discussion on performance



[... truncated for brevity ...]

---

## Issue #N/A: TP4 fails with 5090 in the mix

**Link**: https://github.com/vllm-project/vllm/issues/15576
**State**: open
**Created**: 2025-03-26T21:04:40+00:00
**Comments**: 4
**Labels**: unstale

### Description

> I have a system with a 5090 + 2x4090+ A6000
> 
> I did build as instructions were mentioned, but I have issues when it has to use the 5090 alongside other GPUs.
> 
> Using any pair of 4090 + 4090 or 4090 + A6000 works fine (with TP 2), but when trying to mix the 5090 with any GPU, it fails. So it also happens with TP 4.
> 
> Errors mostly seems to be:
> 
> ```
> (VllmWorkerProcess pid=74788) ERROR 03-23 21:33:26 [multiproc_worker_utils.py:238] Exception in worker VllmWorkerProcess while processing method determine_num_available_blocks.
> (VllmWorkerProcess pid=74788) ERROR 03-23 21:33:26 [multiproc_worker_utils.py:238] Traceback (most recent call last):
> [rank2]:[E323 21:33:26.170490902 ProcessGroupNCCL.cpp:1896] [PG ID 2 PG GUID 3 Rank 2] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
> CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
> For debugging

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM with ray backend and enable nsight can't get perf metrics due to connection issue

**Link**: https://github.com/vllm-project/vllm/issues/7830
**State**: closed
**Created**: 2024-08-24T03:33:30+00:00
**Closed**: 2025-03-04T06:33:14+00:00
**Comments**: 4
**Labels**: bug, ray, unstale

### Description

### Your current environment

<details>
<summary>PyTorch version: 2.3.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.0
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-117-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.5.82
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A100-PCIE-40GB
Nvidia driver version: 555.42.06
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.2.1
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.2.1
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.2.1
/usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9.2.1
/usr/lib/x86_64-linux-gnu/libcudnn_engines_runtime_compiled.s

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Chat inputs to AsyncLLMEngine

**Link**: https://github.com/vllm-project/vllm/issues/14289
**State**: open
**Created**: 2025-03-05T13:25:09+00:00
**Comments**: 3
**Labels**: feature request, unstale

### Description

### 🚀 The feature, motivation and pitch

Currently, only the `LLM` class meant for offline inference supports the `chat` [method](https://docs.vllm.ai/en/latest/models/generative_models.html#llm-chat).
Are there any plans to implement a similar method for `AsyncLLMEngine`, besides the existing `generate`?
Alternatively, is there any work on extending the `PromptType` acceptable by `generate` to include more prompt variants, such as chat conversations?

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: How to use Multi-instance in Vllm? (Model replication on multiple GPUs)

**Link**: https://github.com/vllm-project/vllm/issues/6155
**State**: closed
**Created**: 2024-07-05T14:32:33+00:00
**Closed**: 2025-03-06T12:47:24+00:00
**Comments**: 19
**Labels**: usage, unstale

### Description


I would like to use techniques such as Multi-instance Support supported by the tensorrt-llm backend. In the documentation, I can see that multiple models are served using modes like Leader mode and Orchestrator mode. Does vLLM support this functionality separately? Or should I implement it similarly to the tensorrt-llm backend?

Here is for reference url : https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#leader-mode


---

## Issue #N/A: hidden-states from final (or middle layers)

**Link**: https://github.com/vllm-project/vllm/issues/5406
**State**: closed
**Created**: 2024-06-11T04:06:06+00:00
**Closed**: 2024-12-19T15:14:59+00:00
**Comments**: 4
**Labels**: feature request, unstale

### Description

### 🚀 The feature, motivation and pitch

I am trying to extract hidden states from the final layer of llama3-8b (i.e., the final batch_size, seq_length, n_emb vector _before_ computing the logits). Would it be possible to add this functionality (i.e., access to hidden states similar to transformers ouput_hidden_states)? Thank you!

### Alternatives

HuggingFace Transformers, but this is too slow.

### Additional context

I am trying to train a SAE/linear probe on hidden states from llama3. 

---

## Issue #N/A: [Bug]: Error with structured output inference after upgrade 0.6.2->0.6.3

**Link**: https://github.com/vllm-project/vllm/issues/9462
**State**: open
**Created**: 2024-10-17T12:35:39+00:00
**Comments**: 2
**Labels**: bug, structured-output, unstale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
ollecting environment information...
/opt/conda/lib/python3.11/site-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from vllm.version import __version__ as VLLM_VERSION
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.11.8 | packaged by conda-forge | (main, Feb 16 2024, 20:53:32) [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-6.1.109-118.189.amzn2023.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A10G
Nvidia drive

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: XPU dependencies are missing

**Link**: https://github.com/vllm-project/vllm/issues/11173
**State**: open
**Created**: 2024-12-13T12:03:16+00:00
**Comments**: 11
**Labels**: installation, unstale

### Description

### Your current environment

```text
[W1213 12:52:10.163702538 OperatorEntry.cpp:155] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
    registered at /build/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /build/pytorch/build/aten/src/ATen/RegisterCPU.cpp:30476
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:2971 (function operator())
Collecting environment information...
PyTorch version: 2.5.1+cxx11.abi
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Arch Linux (x86_64)
GCC version: (GCC) 14.2.1 20240910
Clang version: 18.1.8
CMake version: version 3.31.1
Libc ve

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Hidden states processor

**Link**: https://github.com/vllm-project/vllm/issues/12249
**State**: open
**Created**: 2025-01-21T07:09:32+00:00
**Comments**: 19
**Labels**: RFC, unstale

### Description

### Motivation.

Since #10674, vLLM uses Pooler to extract hidden states from the model and convert them to embeddings, class probabilities, and so on. However, this is still not user-friendly enough:

- We have separate model runners for generative and pooling models. This complicates the effort to return hidden states alongside generated text (e.g.: #6165, #11397, #11577, #11606, #11905)
- Setting the default Pooler based on downstream task only covers the common cases. It may be required to use `--override-pooler-config` which isn't that intuitive to use (e.g. #12085). Even so, we still lack support for custom processing of hidden states (e.g. #11065, #11881, #12162)

### Proposed Change.

Similar to `LogitsProcessor` (#1469), we can pass a custom `HiddenStatesProcessor` in `SamplingParams` and `PoolingParams` to postprocess the hidden states and return them in the output. This provides maximum flexibility and enables the same model to be used for different downstream tasks.

```py


[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Docker GPU image is unnecessarily fat due to two (mismatching) copies of CUDA runtime libraries

**Link**: https://github.com/vllm-project/vllm/issues/14433
**State**: open
**Created**: 2025-03-07T10:53:54+00:00
**Comments**: 2
**Labels**: bug, unstale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.35

Python version: 3.12.9 (main, Feb  5 2025, 08:49:00) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-52-generic-x86_64-with-glibc2.35
Is CUDA available: False
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: Could not collect
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        46 bits physical, 48 bits vi

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: RuntimeError: CUDA error: no kernel image is available for execution on the device

**Link**: https://github.com/vllm-project/vllm/issues/5547
**State**: open
**Created**: 2024-06-14T16:13:53+00:00
**Comments**: 22
**Labels**: bug, unstale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```
PyTorch version: 2.3.0
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.5
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-107-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 11.7.99
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: Tesla V100-PCIE-32GB

Nvidia driver version: 535.171.04
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.5.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.5.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.5.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.5.0
/usr/lib/x86_64-linux-gnu/l

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: how to abort request and stop inference?

**Link**: https://github.com/vllm-project/vllm/issues/6975
**State**: closed
**Created**: 2024-07-31T06:49:14+00:00
**Closed**: 2025-03-27T18:35:30+00:00
**Comments**: 6
**Labels**: usage, unstale

### Description

### Your current environment

vllm 0.5.0post1


### How would you like to use vllm

I want to abort a request and stop inference actively, considering the case: an inference of a request lasts too long time or generate token repeatly and cannot stop, I want to stop the inference in vllm (do not need to stop by user)


---

## Issue #N/A: Do vLLM support `input_embeds` as input while using LLama?

**Link**: https://github.com/vllm-project/vllm/issues/8323
**State**: closed
**Created**: 2024-09-10T07:17:49+00:00
**Closed**: 2025-05-02T08:06:40+00:00
**Comments**: 20
**Labels**: bug, unstale

### Description

Can we directly pass the input_embeds to the generate function? Just like the following used in the pytorch transformers

```
generated_ids = self.model.generate(
            inputs_embeds=input_token_embedding,
            do_sample=True,
            max_length=max_length,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_target_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repeat_penalty,
            min_new_tokens=50,
        )
```

---

## Issue #N/A: [New Model]: Support Zyphra/Zamba2-7B

**Link**: https://github.com/vllm-project/vllm/issues/9382
**State**: closed
**Created**: 2024-10-15T16:53:55+00:00
**Closed**: 2025-04-05T11:11:30+00:00
**Comments**: 6
**Labels**: new-model, unstale

### Description

### The model to consider.

Announcement blog: https://www.zyphra.com/post/zamba2-7b

Base model: https://huggingface.co/Zyphra/Zamba2-7B
Instruct tuned: https://huggingface.co/Zyphra/Zamba2-7B-Instruct

![image](https://github.com/user-attachments/assets/bba7f100-f7cf-4284-b8b0-90ed99d9a522)


### The closest model vllm already supports.

Jamba, as it is a mixture of state-space and transformers blocks

> Zamba2-7B-Instruct is a hybrid model composed of state-space ([Mamba2](https://github.com/state-spaces/mamba)) and transformer blocks.

### What's your difficulty of supporting the model you want?

Should be easy once Mamba2 support lands in https://github.com/vllm-project/vllm/pull/9292, however this `use_shared_attention_lora` case seems possibly complex

All of the HF-compatible modeling code can be found here: https://github.com/Zyphra/transformers_zamba2/tree/main/src/transformers/models/zamba2

### Before submitting a new issue...

- [X] Make sure you al

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM does not support virtual GPU

**Link**: https://github.com/vllm-project/vllm/issues/5328
**State**: closed
**Created**: 2024-06-07T00:36:57+00:00
**Closed**: 2025-02-11T16:56:04+00:00
**Comments**: 2
**Labels**: bug, unstale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### 🐛 Describe the bug

error reported by https://github.com/vllm-project/vllm/issues/4587 .

we need to avoid initializing nccl when the world size is 1.

---

## Issue #N/A: [Usage]: mlx-community/DeepSeek-R1-4bit exception：OSError: /data/coding/model-671b-MS/dir does not appear to have a file named configuration_deepseek.py；

**Link**: https://github.com/vllm-project/vllm/issues/13283
**State**: open
**Created**: 2025-02-14T10:55:27+00:00
**Comments**: 5
**Labels**: usage, unstale

### Description

### Your current environment

```text

Use the official vllm repository code：
https://github.com/vllm-project/vllm.git  
cd vllm && pip install -e .  


We downloaded the model from modelscope
total 409717132
-rw-r--r-- 1 root root        761 Feb 14 06:36 README.md
-rw-r--r-- 1 root root       1857 Feb 14 11:51 config.json
-rw-r--r-- 1 root root       1853 Feb 14 11:47 config_bak.json
-rw-r--r-- 1 root root         64 Feb 13 16:36 configuration.json
-rw-r--r-- 1 root root 4139040883 Feb 13 20:11 model-00001-of-00088.safetensors
-rw-r--r-- 1 root root 4845794023 Feb 13 18:23 model-00002-of-00088.safetensors
-rw-r--r-- 1 root root 4697621266 Feb 13 18:32 model-00003-of-00088.safetensors
-rw-r--r-- 1 root root 4845794093 Feb 13 18:20 model-00004-of-00088.safetensors
-rw-r--r-- 1 root root 4845794031 Feb 13 20:16 model-00005-of-00088.safetensors
-rw-r--r-- 1 root root 4697621262 Feb 13 18:30 model-00006-of-00088.safetensors
-rw-r--r-- 1 root root 4845794091 Feb 13 18:22 model-00007-of-0008

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Support structured output and tool call together

**Link**: https://github.com/vllm-project/vllm/issues/16313
**State**: open
**Created**: 2025-04-09T04:33:27+00:00
**Comments**: 2
**Labels**: feature request, unstale

### Description

### 🚀 The feature, motivation and pitch

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as discussed in: https://www.reddit.com/r/LocalLLaMA/comments/1h2y7ys/why_can_you_not_use_structured_output_and_tool/

request support for tool execution followed by structured response output, similar to how OpenAI handles function calls (Tool Calls) and Schema outputs in the gpt-4o-mini API.

OpenAI supports this capability through a workflow where:

The model can first execute one or more tool calls to perform calculations or retrieve information
After receiving the results of these tool calls, the model can then produce a final structured response conforming to a predefined schema

**Example Implementation from OpenAI**

Input to OpenAI
User query: "current age of Univer

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Cannot use model with shorter context as draft model

**Link**: https://github.com/vllm-project/vllm/issues/7859
**State**: open
**Created**: 2024-08-26T04:18:55+00:00
**Comments**: 5
**Labels**: bug, unstale

### Description

### Your current environment

I'm running the latest vllm/vllm-openai docker image on an 8xH100 node

### 🐛 Describe the bug

I'm trying to run vllm with mistral large 2 (123B) and mistral 7B 0.3 as the draft model. However, since the 7B model only has a 32k context to the target models 128K context, I often run into 
raise RuntimeError("Cannot handle cases where distributed draft "
                               "workers generate no tokens")

https://github.com/vllm-project/vllm/blob/0b769992ec1d780b3229c46152c6e647da113aa6/vllm/spec_decode/spec_decode_worker.py#L576
Is there a solution to this?

How to repro:
```
vllm serve mistralai/Mistral-Large-Instruct-2407 --dtype auto --port 8000 --max-model-len 128000 --served-model-name baseten/8w6xo22w --tensor-parallel-size 8  --speculative-model mistralai/Mistral-7B-Instruct-v0.3 --num-speculative-tokens 10 --num-lookahead-slots 10 --use-v2-block-manager --gpu-memory-utilization 0.95 --uvicorn-log-level warning
```


### Before 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: latest docker build (0.6.2) got error due to VLLM_MAX_SIZE_MB

**Link**: https://github.com/vllm-project/vllm/issues/9307
**State**: open
**Created**: 2024-10-12T05:52:08+00:00
**Comments**: 5
**Labels**: bug, unstale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Python version: 3.10.12 (main, Sep 11 2024, 15:47:36) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-113-generic-x86_64-with-glibc2.35
CUDA runtime version: 12.4.131
GPU 0: NVIDIA H20
Nvidia driver version: 555.42.02

```

</details>


### Model Input Dumps

_No response_

### 🐛 Describe the bug

it's a docker build issue,

```sh
DOCKER_BUILDKIT=1 docker build . --tag vllm:0.6.2 
```

error log:

```yml
 => ERROR [build 15/15] RUN if [ "true" = "true" ]; then         python3 check-wheel-size.py dist;     else         echo "Skipping wheel size check  0.4s
------
 > [build 15/15] RUN if [ "true" = "true" ]; then         python3 check-wheel-size.py dist;     else         echo "Skipping wheel size check.";     fi:
0.354 Not allowed: Wheel dist/vllm-0.6.3.dev173+g36

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: No module named `jsonschema.protocols`. 

**Link**: https://github.com/vllm-project/vllm/issues/6486
**State**: open
**Created**: 2024-07-16T23:06:32+00:00
**Comments**: 9
**Labels**: bug, unstale

### Description

### Your current environment

```text
The output of `python collect_env.py`
Collecting environment information...
/home/ubuntu/.local/lib/python3.8/site-packages/torch/cuda/__init__.py:118: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11060). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
  return torch._C._cuda_getDeviceCount() > 0
WARNING 07-16 23:02:53 _custom_ops.py:14] Failed to import from vllm._C with ImportError("/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found (required by /home/ubuntu/.local/lib/python3.8/site-packages/vllm/_C.abi3.so)")
PyTorch version: 2.3.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to b

[... truncated for brevity ...]

---

