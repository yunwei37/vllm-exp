# recent_burst_30days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 21
- Closed Issues: 9

### Label Distribution

- bug: 14 issues
- usage: 7 issues
- feature request: 6 issues
- performance: 1 issues
- RFC: 1 issues
- startup-ux: 1 issues
- installation: 1 issues
- ray: 1 issues

---

## Issue #N/A: [Bug]: Since  #18437 can't serve any Dual Chunked attention model

**Link**: https://github.com/vllm-project/vllm/issues/20484
**State**: closed
**Created**: 2025-07-04T11:41:42+00:00
**Closed**: 2025-07-06T02:38:03+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text

Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version                : 14.0.0-1ubuntu1.1
CMake version                : version 4.0.2
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.9.0.dev20250702+cu128
Is debug build               : False
CUDA used to build PyTorch   : 12.8
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.10.12 (main, May 27 2025, 17:12:29) [GCC 11.4.0] (64-bit runtime)
Python platform              : Linux-5.15.0-105-generic-x86_64-wi

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Benchmark script is not sending requests to serve with the given request rate

**Link**: https://github.com/vllm-project/vllm/issues/21104
**State**: open
**Created**: 2025-07-17T08:26:29+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>
Collecting environment information...                                                                                                                                                   
==============================                                                                                                                                                          
        System Info                                                                                                                                                                     
==============================                                                                                                                                                          
OS                           : CentOS Stream 9 (x86_64)                                                                                                 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Is image embedding supported in llama 4

**Link**: https://github.com/vllm-project/vllm/issues/20993
**State**: open
**Created**: 2025-07-15T15:30:23+00:00
**Comments**: 1
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

I want to run inference of a [Llama-4-Maverick](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8) by directly passing the image embedding, following the instruction [here](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html#embedding-inputs). But I am observing the following error message which appears that the execution is trying to process the image embedding as if it is an image.

After some code search I figured that passing image embedding is [not supported](https://github.com/vllm-project/vllm/blob/68d28e37b0d3706601b0d5231178cebaad032605/vllm/model_executor/models/mllama.py#L1395-L1396) whereas the Llava model in the example does have [type](https://github.com/vllm-project/vllm/blob/e7e3e6d2636f6cd012c7ffeff773b20b3c90b958/vllm/model_executor/mo

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Qwen3 Rerank 模型的准确率存在问题

**Link**: https://github.com/vllm-project/vllm/issues/20478
**State**: closed
**Created**: 2025-07-04T08:28:27+00:00
**Closed**: 2025-07-09T01:23:38+00:00
**Comments**: 23
**Labels**: bug

### Description

### Your current environment

<details>
按照官网下载最新版的vllm pip包，daily的，0.9.2rc版本 main分支
GPU卡为 H20

</details>


### 🐛 Describe the bug

<details>
启动命令：
python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8181 --served-model-name Qwen3-Rerank --model /mnt/data/.t1/dianjin-0701/Qwen3-Reranker-0.6B --task score --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' &> start.log &
curl命令：
curl http://127.0.0.1:8181/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-Rerank",
    "query": "什么是机器学习？",
    "documents": [
      "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式",
      "机器学习是一种编程语言，用于开发网站",
      "机器学习是数据库管理系统的一种",
      "机器学习是操作系统的一种类型"
    ],
    "top_n": 2
  }'
结果：

![Image](https://github.com/user-attachments/assets/7a269fd1-b3fd-49c4-9d07-a7bff619e3f5)
使用bge-rerank-v2-m3版本模型进行同样测试
启动命令：
python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8181 --s

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: rest api multimodal placeholder prompts

**Link**: https://github.com/vllm-project/vllm/issues/20293
**State**: open
**Created**: 2025-07-01T03:13:28+00:00
**Comments**: 3
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Dear vllm community, there is an interest at Spotify in vllm supporting a more flexible multi modal rest api. 

I am interested in being able to prompt vllm with a textual prompt templated with certain placeholders that are to be populated with embed vectors. Today this seems to be supported through [Multimodal inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html) at the vllm library level. 

However it is not accessible at the rest api level, at least this is how I'm understanding the code. Is such support something that is on the roadmap for vllm?

### Alternatives

An alternative I tested out that could work is embed only prompts, which works only on the V0 engine. However ideally it would be good to support text and embeds together.

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of

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

## Issue #N/A: [Feature]: Add Support for Updating Lora Weights

**Link**: https://github.com/vllm-project/vllm/issues/20149
**State**: open
**Created**: 2025-06-26T20:18:59+00:00
**Comments**: 2
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

We are using TRL to train an updated version of one of our models which has a modality specific LoRA adapter (i.e., same as granite speech, phi4mm). TRL does have support for integrating into vLLM, but the way it handles adapters doesn't work effectively for this sort of model, because it assumes the lora weights can be merged (e.g., [here](https://github.com/huggingface/trl/blob/79ec242aefedc108de9edbea62be6d95070fde03/trl/trainer/grpo_trainer.py#L924)). 

As far as I know, the way we would currently 'update' the adapter is to save it out and then reload it, e.g., using the [worker lora manager](https://github.com/vllm-project/vllm/blob/main/vllm/lora/worker_manager.py#L86). It would be nice to have a supported way of updating the LoRA tensors being trained without exporting them though.

If this is possible already, that would be great! Otherwise happy to take a pass at contributing it.  @jeejeelee @avishaiElmakies

### Alternatives

_No respo

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Error when loading EAGLE3 weight, yuhuili/ EAGLE3-LLaMA3.1-Instruct-8B

**Link**: https://github.com/vllm-project/vllm/issues/19991
**State**: open
**Created**: 2025-06-23T16:30:22+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

Docker image: rocm/vllm:rocm6.4.1_vllm_0.9.0.1_20250605


### 🐛 Describe the bug

I'm trying to use vLLM eagle3, using the server launch command as below:
`vllm serve /models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/ --trust-remote-code --swap-space 16 --disable-log-requests --tensor-parallel-size 1 --distributed-executor-backend mp --dtype float16 --quantization fp8 --kv-cache-dtype fp8 --no-enable-chunked-prefill --max-num-seqs 300  --max-num-batched-tokens 131072 --gpu-memory-utilization 0.8 --enforce-eager --speculative_config '{"method": "eagle", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 5, "draft_tensor_parallel_size": 1, "dtype": "float16"}'`

For eagle3 model, I used "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", while there is an error of weight shape misalignment as below:

ERROR 06-22 15:12:27 [engine.py:458] Attempted to load weight (torch.Size([4096, 12288])) into para

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Hermes tool call parser: pops empty list

**Link**: https://github.com/vllm-project/vllm/issues/20991
**State**: open
**Created**: 2025-07-15T13:55:52+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

4xH100-80GiB

CLI Args:
```
        # Inference
        - --model
        - Qwen/Qwen3-235B-A22B-FP8
        - --gpu-memory-utilization
        - "0.90"
        - --disable-custom-all-reduce

        - --rope-scaling.rope_type
        - "yarn"
        - --rope-scaling.factor
        - 4
        - --rope-scaling.original_max_position_embeddings
        - 32768
        - --max-model-len
        - "131072"
        - --tensor-parallel-size
        - "4"

        # Function calling
        - --enable-auto-tool-choice
        - --tool-call-parser
        - hermes

        # Server
        - --host
        - "0.0.0.0"
        - --disable-log-requests
```


### 🐛 Describe the bug

Pops from an empty list in the Hermes tool call parser.
```
IndexError: pop from empty list
  File "/usr/local/lib/python3.12/dist-packages/partial_json_parser/core/myelin.py", line 50, in fix_fast
    _i, _char = stack.pop()

  File "/usr/local/lib/python3.12/dist-packages/partial_json_

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: I cannot compile vllm on RTX5090

**Link**: https://github.com/vllm-project/vllm/issues/20345
**State**: open
**Created**: 2025-07-02T01:10:24+00:00
**Comments**: 3
**Labels**: usage

### Description

### Your current environment

我的服务器配置是ubuntu20，conda使用的环境是python 3.12，我的安装命令如下：
```text
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.8.3
python use_existing_torch.py
pip install -r requirements/build.txt
pip install setuptools_scm
```
但是他却报错了：
```text
 /home/pc/data/envs/vllmlast/lib/python3.12/site-packages/setuptools/_distutils/dist.py:1021: _DebuggingTips: Problem in editable installation.
  !!

          ********************************************************************************
          An error happened while installing `vllm` in editable mode.

          The following steps are recommended to help debug this problem:

          - Try to install the project normally, without using the editable mode.
            Does the error still persist?
            (If it does, try fixing the problem before attempting the editable mode).
          - If you are using binary extensions, make sure you have all OS-level
            dependencies installed (e.g. 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Cache EngineCoreOutput for system prompt to prevent repeated calculation

**Link**: https://github.com/vllm-project/vllm/issues/20044
**State**: open
**Created**: 2025-06-24T22:18:36+00:00
**Comments**: 3
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

My company is prefixing a common system prompt to all requests we serve through vLLM. Currently, when we call add_request, we send a request with prompt_token_ids including the entire system prompt. Every time we schedule a request, we are forced to schedule these common system prompt tokens over and over again. 

The RFC I propose is, instead of passing the system prompt tokens per-request, we add a function add_system_prompt which schedules a dummy request whose prompt solely consists of the system prompt. We cache the resulting EngineCoreOutputs. Finally, when we start serving requests, we simply set num_computed_tokens to num_tokens_in_sys_prompt // block_size * block_size. To prevent evicting the system prompt KVCacheBlocks, we add a 'permanent' flag to the KVCacheBlock data structure, and set this flag in the add_system_prompt function.

This feature would significantly help our inference for short context window (max_model_len=2048).

We 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vllm config.py报错selected_task=next(iter(supported_tasks_lst))

**Link**: https://github.com/vllm-project/vllm/issues/21111
**State**: open
**Created**: 2025-07-17T09:42:17+00:00
**Comments**: 1
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

vllm 0.6.4.post1
使用cosyvoice2报错selected_task=next(iter(supported_tasks_lst))
StopIteration

![Image](https://github.com/user-attachments/assets/3961ca6f-780b-44c2-b920-9f7a30201f89)

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: whether torchrun-compatible mode supports DP/EP?

**Link**: https://github.com/vllm-project/vllm/issues/20416
**State**: open
**Created**: 2025-07-03T06:47:37+00:00
**Comments**: 0
**Labels**: usage

### Description

### Your current environment

None

### How would you like to use vllm

I have read https://github.com/vllm-project/vllm/pull/12071 and it's a wonderful work.

I wonder if this torchrun-compatible executor supports EP? Since the comments in https://github.com/vllm-project/vllm/pull/12071 point out that the input should be same across all ranks (maybe the context is TP).

In EP scenario, all ranks in the same EP group should have different input to take the advantage of EP MoE. And if DeepEP is enabled, prefill and decode would dispatch to normal kernels and ll kernels separately. This requires schedulers ascross ranks in the same EP group should schedule the same prefill/decode action with different inputs. Are we now ensuring this behavior or this is not necessary in current design?

### Before submitting a new issue...

- [ ] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Use LoRA on MoE models

**Link**: https://github.com/vllm-project/vllm/issues/20161
**State**: closed
**Created**: 2025-06-27T04:00:16+00:00
**Closed**: 2025-06-27T06:42:41+00:00
**Comments**: 3
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Hi, I would like to know if the vLLM team has plans to support LoRA on the MoE model?
I'm currently trying to get this working but don't know if its really necessary.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: --max-model-len doesn't work

**Link**: https://github.com/vllm-project/vllm/issues/20304
**State**: closed
**Created**: 2025-07-01T06:57:09+00:00
**Closed**: 2025-07-08T07:48:12+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
INFO 06-30 23:25:24 [__init__.py:244] Automatically detected platform cuda.
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version                : Could not collect
CMake version                : version 4.0.2
Libc version                 : glibc-2.35

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
Python version               : 3.12.11 (main, Jun  4 2025, 08:56:18) [GCC 11.4.0] (64-bit runtime)
Py

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

## Issue #N/A: [Feature]: Support for Hunyuan-A13B-Instruct

**Link**: https://github.com/vllm-project/vllm/issues/20182
**State**: closed
**Created**: 2025-06-27T11:36:33+00:00
**Closed**: 2025-06-27T15:59:44+00:00
**Comments**: 2
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Tencent released this new model:
https://huggingface.co/tencent/Hunyuan-A13B-Instruct

It matches bigger models on benchmarks. It has a decent size to run locally and the MoE architecture should make it pretty fast.
It has 256K context too.

The tencent team released a docker version compatible with vllm 0.8.5 but that image lacks the new improvements. Plus I think it doesn't have the Ampere fp8 marlin support as I can't run the fp8 quant it on a 3090 system

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Installation]: Solved Bug: CMake `execute_process` fails to read `/proc/cpuinfo` without absolute path

**Link**: https://github.com/vllm-project/vllm/issues/20458
**State**: open
**Created**: 2025-07-03T23:19:58+00:00
**Comments**: 2
**Labels**: installation

### Description

### Your current environment

```text
Ubuntu, the most important is that  all is standard here: 
 
which cat
/bin/cat
$ file /bin/cat
/bin/cat: ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, BuildID[sha1]=70bb40952afe7016b06511e5c96e926f1f4774ba, for GNU/Linux 3.2.0, stripped

```


I have quickly searched for `cpuinfo` problem in all issues, including closed ones and it seems to be new one. Gemini CLI solved it and wrote most of below: 

### Bug: CMake `execute_process` fails to read `/proc/cpuinfo` without absolute path

**Description:**
When building vLLM with `VLLM_TARGET_DEVICE=cpu`, the CMake configuration fails because the `execute_process` command in `cmake/cpu_extension.cmake` is unable to read `/proc/cpuinfo`. The error message reported is "Failed to check CPU features via /proc/cpuinfo".

**Reproduction Steps:**
1. Clone the vLLM repository.
2. Run `cmake . -DVLLM_TARGET_DEVICE=cpu -D VLLM_PYTHON_EXECUTAB

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Disable the http request log for /metrics

**Link**: https://github.com/vllm-project/vllm/issues/21001
**State**: open
**Created**: 2025-07-15T17:10:47+00:00
**Comments**: 0
**Labels**: usage

### Description

### Your current environment

vLLM 0.9.2
Docker
Prometheus + Grafana
Windows 11 (Also an Ubuntu Server)


### How would you like to use vllm

Hello,

I am using the latest release 0.9.2 with Prometheus + Grafana (with Docker). For the dashboard and related configs, I am using the official documentation example. I have two questions:

1. How can I disable the logging of the HTTP requests for the /metrics endpoint only? Because it is cluttering the entire log and making things hard to follow. I still want to see the log of the other http endpoints.

2. I am able to see that Prometheus is running fine, and I can connect to it from Grafana successfully, but using the official Grafana dashboard, it displays 'No Data' even though there are requests made to the server. Is the dashboard JSON template up-to-date?

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](htt

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: The same query yields different results across different vLLM versions under identical reasoning.

**Link**: https://github.com/vllm-project/vllm/issues/21096
**State**: open
**Created**: 2025-07-17T06:16:02+00:00
**Comments**: 5
**Labels**: bug

### Description

### Your current environment

I've tried three different version of vllm: 0.6.5, 0.8.4, 0.9.2. For example, in the environment, I only ran `pip install vllm==0.6.5`.
The output after running the `collect_env.py`:
```
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version                : Could not collect
CMake version                : Could not collect
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.7.0+cu126
Is debug build               : False
CUDA used to build PyTorch   : 12.6
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.11 | packaged by Anaconda, Inc. | (ma

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Why does the Prefix cache hit rate reach 60% for random data during benchmark?

**Link**: https://github.com/vllm-project/vllm/issues/20015
**State**: open
**Created**: 2025-06-24T09:53:42+00:00
**Comments**: 14
**Labels**: usage

### Description

### Your current environment

```text
root@llm206:/workspace/vllm# python3 ./vllm/collect_env.py
INFO 06-24 09:50:15 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
/usr/local/lib/python3.10/dist-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.3 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version                : Could not collect
CMake version                : version 3.22.1
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.6.0+cu124
Is debug build               : False
CUDA used to build PyTorch   : 12.4
ROCM used to build PyTorch   : N/A

==============================
  

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: 0.9.1(V1) Hanging(RayChannelTimeoutError ) when inferencing guided_json in DeepSeek-R1/V3 (TP=8, PP=2)

**Link**: https://github.com/vllm-project/vllm/issues/21022
**State**: closed
**Created**: 2025-07-16T02:07:16+00:00
**Closed**: 2025-07-16T06:38:07+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
==============================
      Python Environment
==============================
Python version               : 3.10.12 (main, May 27 2025, 17:12:29) [GCC 11.4.0] (64-bit runtime)
Python platform              : Linux-5.10.134-008.18.kangaroo.al8.x86_64-x86_64-with-glibc2.35

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : True
CUDA runtime version         : 12.5.40
CUDA_MODULE_LOADING set to   : LAZY
GPU models and configuration : 
GPU 0: NVIDIA H20
GPU 1: NVIDIA H20
GPU 2: NVIDIA H20
GPU 3: NVIDIA H20
GPU 4: NVIDIA H20
GPU 5: NVIDIA H20
GPU 6: NVIDIA H20
GPU 7: NVIDIA H20

Nvidia driver version        : 550.54.15
cuDNN version                : Could not collect
HIP runtime version          : N/A
MIOpen runtime version       : N/A
Is XNNPACK available         : True

==============================


[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Optimize vectorization utils for `csrc/quantization/vectorization_utils.cuh`

**Link**: https://github.com/vllm-project/vllm/issues/20327
**State**: closed
**Created**: 2025-07-01T18:13:16+00:00
**Closed**: 2025-07-04T07:06:26+00:00
**Comments**: 0
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Currently, the `vectorize_with_alignment` could handle arbitrary elements, but this could also cause some overhead.
To further improve the performance, we can add some branches to it, so eg 
```c++

  // Fast path identical to the one added for the write version above.
  bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    vec_op
  }

  // arbitrary num of elements supported logic now
  ...
```

I will have a pr for this soon

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: Inconsistent behavior of AsyncLLMEngine.abort between v0 and v1

**Link**: https://github.com/vllm-project/vllm/issues/20362
**State**: open
**Created**: 2025-07-02T08:16:50+00:00
**Comments**: 2
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

When using vllm as inference engine for RLHF, we rely on LLMEngine.abort_request or AsyncLLMEngine.abort to stop long running requests.

The usage of AsyncLLMEngine.abort can be found in test https://github.com/vllm-project/vllm/blob/v0.9.1/tests/async_engine/test_async_llm_engine.py#L350 and https://github.com/vllm-project/vllm/blob/main/tests/v1/engine/test_async_llm.py#L185. 

When using v0 async engine, we can abort a request from other coroutine and the generate coroutine will raise asyncio.CancelledError.

When using v1 async engine, we have to cancel the generation coroutine itself and if we abort the request but not cancel
the generation coroutine, it will never return which is not expected and unfriendly for programming.

So, my question is, can we change the behavior of AsyncL

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]:

**Link**: https://github.com/vllm-project/vllm/issues/20314
**State**: closed
**Created**: 2025-07-01T10:21:12+00:00
**Closed**: 2025-07-01T10:25:19+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
NFO 07-01 10:11:23 [__init__.py:243] Automatically detected platform cuda.
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.4 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version                : Could not collect
CMake version                : version 3.30.2
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.7.0+cu126
Is debug build               : False
CUDA used to build PyTorch   : 12.6
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0] (64-bit runtime)
Py

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: vLLM Server Launch Freezes at Using NCCL on B200

**Link**: https://github.com/vllm-project/vllm/issues/20862
**State**: open
**Created**: 2025-07-12T20:31:16+00:00
**Comments**: 20
**Labels**: usage

### Description

### Your current environment

```text
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version                : Could not collect
CMake version                : version 4.0.2
Libc version                 : glibc-2.35

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
Python version               : 3.12.11 (main, Jun  4 2025, 08:56:18) [GCC 11.4.0] (64-bit runtime)
Python platform              : Linux-5.15.0-143-generic-x86_64-with-glibc2.35

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : 

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: How to implement request abortion in vLLM OpenAI API Server?​

**Link**: https://github.com/vllm-project/vllm/issues/20798
**State**: open
**Created**: 2025-07-11T05:30:18+00:00
**Comments**: 3
**Labels**: usage

### Description

### Your current environment

```text
 I’m deploying vLLM’s OpenAI-compatible API server (python -m vllm.entrypoints.openai.api_server) for real-time inference. In production, users may need to ​​cancel long-running requests​​ (e.g., closing a chat session). However, the official documentation lacks details on interrupting ongoing inference tasks via the API.

Can I implement an endpoint (e.g., POST /v1/completions/{request_id}/cancel) to ​​terminate a specific inference request​​ mid-generation, freeing GPU resources immediately？

please give me some advice
```


### How would you like to use vllm

I want to abort a request inference of a [qwen3:8b]. I don't know how to do with the api with vllm.


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: v0.9.1 - ignoring the input arguments to engine

**Link**: https://github.com/vllm-project/vllm/issues/20241
**State**: closed
**Created**: 2025-06-30T06:13:22+00:00
**Closed**: 2025-07-13T05:08:54+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```
==============================
         vLLM Info
==============================
ROCM Version                 : Could not collect
Neuron SDK Version           : N/A
vLLM Version                 : 0.9.1
vLLM Build Flags:
  CUDA Archs: 8.0;8.6;9.0; ROCm: Disabled; Neuron: Disabled

==============================
       PyTorch Info
==============================
PyTorch version              : 2.7.0+cu126
Is debug build               : False
CUDA used to build PyTorch   : 12.6
ROCM used to build PyTorch   : N/A

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : True
CUDA runtime version         : 12.6.85
CUDA_MODULE_LOADING set to   : LAZY
GPU models and configuration :
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3
GPU 4: NVIDIA H100 80GB HBM3


[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: PD does not work with ray distributed backend

**Link**: https://github.com/vllm-project/vllm/issues/21070
**State**: open
**Created**: 2025-07-16T18:34:27+00:00
**Comments**: 0
**Labels**: bug, ray

### Description

### Your current environment

<details>
<summary> run vllm.sh which uses ray as the backend </code></summary>

```text
#!/bin/bash
set -xe

# Models to run
MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  # "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
)

export VLLM_LOGGING_LEVEL=debug
# export NIXL_LOG_LEVEL=DEBUG
# export UCX_LOG_LEVEL=trace

# Number of prefill and decode instances to create
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1} # Default to 1
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}   # Default to 2

# Find the git repository root directory
# GIT_ROOT=$(git rev-parse --show-toplevel)

SMI_BIN=$(which nvidia-smi || which rocm-smi)

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

# Waits for vLLM to start.
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# # Function to clean up previous i

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Specifying Medusa Choice Tree in vllm"

**Link**: https://github.com/vllm-project/vllm/issues/20813
**State**: open
**Created**: 2025-07-11T11:15:27+00:00
**Comments**: 2
**Labels**: usage

### Description

### How would you like to use vllm

**Description**
I'm using `vllm` to load a model with a Medusa heads. My current implementation uses the following setup:

```python
from vllm import SamplingParams
from vllm import EngineArgs, LLMEngine

MODEL_NAME = "JackFram/llama-68m"
SPEC_MODEL = "abhigoyal/vllm-medusa-llama-68m-random"

llm = LLM(
    model=MODEL_NAME,
    max_model_len=1024,
    speculative_config={
        "method" : "medusa",
        "model": SPEC_MODEL,
        "num_speculative_tokens": 3,
    },
    tensor_parallel_size=1,
    seed=0,
)
outputs = llm.generate(prompts=["Hi! How are you doing?", "Hi! How are you doing?"], use_tqdm=True)
```

Question
I want to know how to specify the Medusa choice tree for the model. Could you provide guidance or examples on how to do this?

Environment

- Python version: 3.11
- vllm version: 0.9.2
- OS: linux

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the 

[... truncated for brevity ...]

---

