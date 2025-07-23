# power_users_over20 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 2
- Closed Issues: 28

### Label Distribution

- bug: 11 issues
- stale: 6 issues
- feature request: 5 issues
- RFC: 2 issues
- new-model: 2 issues
- usage: 2 issues
- help wanted: 1 issues
- misc: 1 issues
- unstale: 1 issues
- rocm: 1 issues

---

## Issue #N/A: [Feature]: Implement Priority Scheduling In V1 Engine

**Link**: https://github.com/vllm-project/vllm/issues/14002
**State**: closed
**Created**: 2025-02-28T01:33:35+00:00
**Closed**: 2025-06-23T03:18:09+00:00
**Comments**: 10
**Labels**: help wanted, feature request

### Description

### 🚀 The feature, motivation and pitch

In V0, we support request priority. I would like to see this in V1

cc @WoosukKwon 

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Benchmark: benchmark_throughput and benchmark_latency should be able to write output to JSON file. 

**Link**: https://github.com/vllm-project/vllm/issues/4847
**State**: closed
**Created**: 2024-05-16T02:21:56+00:00
**Closed**: 2024-05-16T17:02:57+00:00
**Comments**: 1
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Similar to `benchmarks/benchmark_serving.py`, the throughput and latency benchmark should be able to write their metrics to JSON file for result aggregated so we don't need to parse the log data. 

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: Mixtral AWQ fails to work: asyncio.exceptions.CancelledError: Cancelled by cancel scope 7fd214489990

**Link**: https://github.com/vllm-project/vllm/issues/2621
**State**: closed
**Created**: 2024-01-27T01:26:26+00:00
**Closed**: 2024-04-04T15:15:39+00:00
**Comments**: 12

### Description

```
export CUDA_HOME=/usr/local/cuda-12.3
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu123"
pip install git+https://github.com/vllm-project/vllm.git --upgrade
export CUDA_VISIBLE_DEVICES=1

python -m vllm.entrypoints.openai.api_server --port=5002 --host=0.0.0.0 --model TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ --quantization awq --dtype auto --seed 1234 --tensor-parallel-size=1 --max-num-batched-tokens=66560 --max-log-len=100
```

Any where, even simple, leads to:

```
INFO 01-27 01:15:31 api_server.py:209] args: Namespace(host='0.0.0.0', port=5002, allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, served_model_name=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, root_path=None, middleware=[], model='TheBloke/Mixtral-8x7B-Instru>
WARNING 01-27 01:15:31 config.py:176] awq quantization is not fully optimized yet. The speed can be slower than non-quantized model

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Inline Golden (Expected) Tests

**Link**: https://github.com/vllm-project/vllm/issues/4663
**State**: closed
**Created**: 2024-05-07T23:47:12+00:00
**Closed**: 2024-11-28T02:05:17+00:00
**Comments**: 6
**Labels**: RFC, stale

### Description

### Motivation.

Some of tests in vllm is neither sufficient nor easy to read, e.g.

https://github.com/vllm-project/vllm/blob/8344f7742b794ca6ec9bcb891c178cd0551f23d0/tests/core/test_scheduler.py#L293-L297

The test `assert out.blocks_to_swap_out != {}` is insufficient, and these lines only test certain properties of the output.

### Proposed Change.

We can use inline golden tests (a.k.a. expected tests) from https://github.com/ezyang/expecttest, which is used heavily in pytorch:

https://github.com/pytorch/pytorch/blob/8b4d62009ddbc24a69dfcdbebc2cc84e4b2ee8f5/test/test_python_dispatch.py#L645-L654

### Feedback Period.

_No response_

### CC List.

_No response_

### Any Other Things.

_No response_

---

## Issue #N/A: [Bug]: glm4v Is Broken

**Link**: https://github.com/vllm-project/vllm/issues/14529
**State**: closed
**Created**: 2025-03-10T02:14:07+00:00
**Closed**: 2025-03-13T11:37:19+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

Per comment

### 🐛 Describe the bug

- GLM4V is broken on V0 and V1

```bash
VLLM_USE_V1=0 pytest -v -x models/decoder_only/vision_language/test_models.py -k glm4v
```

```bash
def _validate_mm_placeholders(
        self,
        mm_placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_item_counts: Mapping[str, int],
    ) -> None:
        for modality, item_count in mm_item_counts.items():
            placeholders = mm_placeholders.get(modality, [])
    
            if len(placeholders) != item_count:
>               raise RuntimeError(
                    f"Expected there to be {item_count} prompt updates "
                    f"corresponding to {item_count} {modality} items, but "
                    f"instead found {len(placeholders)} prompt updates! "
                    "Either the prompt text has missing/incorrect tokens for "
                    "multi-modal inputs, or there is a problem with your "
                    "implementati

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: offline test, Process hangs without exiting when using cuda graph

**Link**: https://github.com/vllm-project/vllm/issues/4263
**State**: closed
**Created**: 2024-04-22T09:53:12+00:00
**Closed**: 2024-05-09T12:37:47+00:00
**Comments**: 17
**Labels**: bug

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.2.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.2
Libc version: glibc-2.35

Python version: 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-4.18.0-240.el8.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.3.107
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L20
GPU 1: NVIDIA L20
GPU 2: NVIDIA L20
GPU 3: NVIDIA L20
GPU 4: NVIDIA L20
GPU 5: NVIDIA L20
GPU 6: NVIDIA L20
GPU 7: NVIDIA L20

Nvidia driver version: 550.54.15
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.0.0
/usr/lib/x86_64-linux

[... truncated for brevity ...]

---

## Issue #N/A: There are often recomputations in the prefill stage, is that a bug?

**Link**: https://github.com/vllm-project/vllm/issues/812
**State**: closed
**Created**: 2023-08-21T11:04:49+00:00
**Closed**: 2024-05-31T04:19:56+00:00
**Comments**: 2

### Description

Here is my code in this file:  vllm/worker/worker.py
```python
    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> Dict[int, SequenceOutputs]:

        print(f"begin to run {len(seq_group_metadata_list)} groups")
        num = len(seq_group_metadata_list)
        need_prefill = False
        need_decoding = False
        if num > 1:
            for seq in seq_group_metadata_list:
                for k, v in seq.seq_data.items():
                    if len(v.output_token_ids) == 0:
                        need_prefill = True
                    else:
                        need_decoding = True
        if need_prefill == True and need_decoding == True:
            import pdb; pdb.set_trace()
```
I want to capture such situation: a batched input with

[... truncated for brevity ...]

---

## Issue #N/A: Improve Weight Loading

**Link**: https://github.com/vllm-project/vllm/issues/48
**State**: closed
**Created**: 2023-04-22T04:16:22+00:00
**Closed**: 2023-05-03T07:32:05+00:00
**Comments**: 0

### Description

Just use Huggingface's weights. Don't do another copy!

---

## Issue #N/A: [Misc]: setting environment variables in multi-node serving

**Link**: https://github.com/vllm-project/vllm/issues/6803
**State**: closed
**Created**: 2024-07-25T21:55:21+00:00
**Closed**: 2024-07-25T22:38:33+00:00
**Comments**: 0
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

As we embrace large models like Llama 3.1 405B, lots of users are trying multi-node inference now.

Compared with single-node inference, multi-node inference is much more difficult to set up, due to the nature of complicated machine configuration.

The [documentation](https://docs.vllm.ai/en/stable/serving/distributed_serving.html#multi-node-inference-and-serving) serves as a starting point. And we have a [debugging guide](https://docs.vllm.ai/en/stable/getting_started/debugging.html) with a sanity check script for testing the configuration. Even with this help, users might still find it difficult to set up the cluster, especially w.r.t. network configuration to make the machines talk to each other.

This discussion issue tries to clarify one aspect: how to set environment variables in multi-node setting, and how does environment variable inheritance work  in multi-node serving.

> NOTE: https://github.com/vllm-project/vllm/issues/6

[... truncated for brevity ...]

---

## Issue #N/A: vLLM computes max sequence length for Yi 200k at 4k

**Link**: https://github.com/vllm-project/vllm/issues/2565
**State**: closed
**Created**: 2024-01-23T21:23:40+00:00
**Closed**: 2024-01-23T21:32:16+00:00
**Comments**: 2

### Description

https://huggingface.co/01-ai/Yi-34B-Chat/blob/main/config.json

vLLM not accounting for rope scaling.

So can't use full context.

```
 -m vllm.entrypoints.openai.api_server \
        --port=5000 \
        --host=0.0.0.0 \
        --model=01-ai/Yi-34B-Chat \
        --seed 1234 \
        --tensor-parallel-size=4 \
        --trust-remote-code \
        --max-model-len=204800 \
        --download-dir=/workspace/.cache/huggingface/hub
```

gives:
```
INFO 01-23 21:18:03 api_server.py:727] args: Namespace(host='0.0.0.0', port=5000, allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], served_model_name=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, model='01-ai/Yi-34B-Chat', tokenizer=None, revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=True, download_dir='/workspace/.cache/huggingface/hub', load_format='auto', dtype='auto', max_model_len=204800, worker_u

[... truncated for brevity ...]

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

## Issue #N/A: [Feature]: ci test with vGPU

**Link**: https://github.com/vllm-project/vllm/issues/5426
**State**: closed
**Created**: 2024-06-11T16:39:05+00:00
**Closed**: 2024-11-27T02:07:12+00:00
**Comments**: 4
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

it seems aws and gcp supports [vGPU](https://docs.nvidia.com/grid/cloud-service-support.html) . we can run some small tests in vGPU, which should be cost-efficient and also test broader software support to avoid https://github.com/vllm-project/vllm/issues/4587 .

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [Bug]: Dynamic FP8 Marlin quantization fails on `0.5.4` 

**Link**: https://github.com/vllm-project/vllm/issues/7216
**State**: closed
**Created**: 2024-08-06T19:33:19+00:00
**Closed**: 2024-08-07T18:23:13+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

<details>

```
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.2
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-107-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.5.82
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB
GPU 2: NVIDIA A100-SXM4-80GB
GPU 3: NVIDIA A100-SXM4-80GB
GPU 4: NVIDIA A100-SXM4-80GB
GPU 5: NVIDIA A100-SXM4-80GB
GPU 6: NVIDIA A100-SXM4-80GB
GPU 7: NVIDIA A100-SXM4-80GB

Nvidia driver version: 550.54.15
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK av

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Dose V1  support MLA + PP now? Raise error while using PP+TP+V1.

**Link**: https://github.com/vllm-project/vllm/issues/14263
**State**: closed
**Created**: 2025-03-05T07:03:58+00:00
**Closed**: 2025-03-12T12:02:02+00:00
**Comments**: 1
**Labels**: usage

### Description

Device:

3xL20x8, DeepSeek-R1, PP=3, TP=8, VLLM_USE_V1=1

Launch:

```bash
python3 -m vllm.entrypoints.openai.api_server  --dtype=auto --tensor-parallel-size=8 --host=0.0.0.0 --port=80 --tokenizer-mode=slow --model=/DeepSeek-R1 --block-size=32 --swap-space=16 --g
pu-memory-utilization=0.9 --pipeline-parallel-size=3 --max-num-seqs=48 --trust-remote-code --no-enable-prefix-caching  --enable-chunked-prefill=True --max-model-len=16384 --max-num-batched-tokens=2048 --served-model-name=/DeepSeek-R1
```

Error:
```
[2025-03-05 14:28:30,815] [INFO] [MainThread] [vllm.transformers_utils.config] >>> Replacing legacy 'type' key with 'rope_type'
[2025-03-05 14:28:34,452] [INFO] [MainThread] [vllm.platforms] >>> Automatically detected platform cuda.
[2025-03-05 14:28:37,141] [INFO] [MainThread] [vllm.config] >>> This model supports multiple tasks: {'generate', 'embed', 'classify', 'score', 'reward'}. Defaulting to 'generate'.
[2025-03-05 14:28:37,343] [INFO] [MainThread] [vllm.config] >>> Defaultin

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: debugging guide for device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "../aten/src/ATen/cuda/CUDAContext.cpp"

**Link**: https://github.com/vllm-project/vllm/issues/6056
**State**: closed
**Created**: 2024-07-02T05:53:06+00:00
**Closed**: 2024-07-03T06:37:30+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### 🐛 Describe the bug

This is a compond and annoying bug, coupled with pytorch bug https://github.com/pytorch/pytorch/pull/122815 .

Basically, pytorch `torch.cuda.device_count` function will cache the device count when first called. Users might not call it directly, but if you use `import torch._dynamo` , it will be called. The call chain is:

```text
  File "/usr/local/lib/python3.10/dist-packages/torchvision/ops/roi_align.py", line 4, in <module>
    import torch._dynamo
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/usr/local/lib/python3.10/dist-packages/torch/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vllm suffer extreme latency or even timeout when enabling lora

**Link**: https://github.com/vllm-project/vllm/issues/18217
**State**: open
**Created**: 2025-05-15T17:47:01+00:00
**Comments**: 6
**Labels**: bug

### Description

### Your current environment

current environment:
vllm version: `0.8.5.post1`
```
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.2 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.39

Python version: 3.11.11 (main, Dec 11 2024, 16:28:39) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.11.0-25-generic-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 12.0.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA RTX 5000 Ada Generation
GPU 1: NVIDIA RTX 5000 Ada Generation

Nvidia driver version: 550.144.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][ROCm]: Performance issue in ROCm Triton FlashAttention

**Link**: https://github.com/vllm-project/vllm/issues/4018
**State**: closed
**Created**: 2024-04-11T20:47:59+00:00
**Closed**: 2024-09-04T14:05:58+00:00
**Comments**: 4
**Labels**: bug, rocm

### Description

### Your current environment

```text
PyTorch version: 2.1.1+git011de5c
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.0.32830-d62f6a171

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.0 23483 7208e8d15fbf218deb74483ea8c549c67ca4985e)
CMake version: version 3.29.1
Libc version: glibc-2.31

Python version: 3.9.18 (main, Sep 11 2023, 13:41:44)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.19.0-45-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 10.1.243
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI250X/MI250NoGCNArchNameOnOldPyTorch
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: 6.0.32830
MIOpen runtime version: 3.0.0
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU o

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: distribute the package on macos

**Link**: https://github.com/vllm-project/vllm/issues/15661
**State**: open
**Created**: 2025-03-28T01:51:30+00:00
**Comments**: 4
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

Now that vLLM can be built directly in macos with `pip install -e .` , we can build the wheel and publish it to pypi, so that users can directly install it.

This is mostly for testing and development, so that people can develop some pure python code without spinning up a gpu server. This will help lower the barrier of developing vllm.

Along with it, we should also have some nightly ci to have basic smoke test to make sure it is not broken.

cc @simon-mo for release, and @khluu for ci.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: [V1] Beam Search Test Fails on V1

**Link**: https://github.com/vllm-project/vllm/issues/14587
**State**: closed
**Created**: 2025-03-11T00:30:02+00:00
**Closed**: 2025-07-11T02:15:57+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

Per note. It would be great if someone could look into this!

### 🐛 Describe the bug

```bash
pytest VLLM_USE_V1=1 tests/samplers/test_beam_search.py
```

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [CI Failure]: Samplers Test - samplers/test_beam_search.py::test_beam_search_passes_multimodal_data

**Link**: https://github.com/vllm-project/vllm/issues/19736
**State**: closed
**Created**: 2025-06-17T09:23:35+00:00
**Closed**: 2025-06-18T22:48:30+00:00
**Comments**: 0
**Labels**: ci-failure

### Description

### Name of failing test

`samplers/test_beam_search.py::test_beam_search_passes_multimodal_data[False-2-64-half]`

### Basic information

- [ ] Flaky test
- [x] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

It seems the issue is because we are now passing empty lists to _flatten_embeddings

```
FAILED samplers/test_beam_search.py::test_beam_search_passes_multimodal_data[False-2-64-half] - RuntimeError: torch.cat(): expected a non-empty list of Tensors
```

Full output:
```
pytest -s -v "samplers/test_beam_search.py::test_beam_search_passes_multimodal_data[False-2-64-half]"
INFO 06-17 09:19:56 [__init__.py:244] Automatically detected platform cuda.
/home/mgoin/venvs/vllm/lib/python3.12/site-packages/pytest_asyncio/plugin.py:208: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope.

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: improve distributed backend selection

**Link**: https://github.com/vllm-project/vllm/issues/8683
**State**: closed
**Created**: 2024-09-20T22:54:29+00:00
**Closed**: 2024-09-29T01:17:08+00:00
**Comments**: 4
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

We have three ways to start a new process:

- multiprocessing by `fork`
- multiprocessing by `spawn`
- ray

by default, we use ray for multi-node serving, and multiprocessing by `fork` for single node setting.

however, if users initialize cuda context, multiprocessing by `fork` will not work.

if we set multiprocessing by `spawn` by default, it will not work when users don't have `if __name__ == "__main__"`. 

if we can figure out whether users have `if __name__ == "__main__"` automatically, we can improve the default user experience.

the proposed solution is:

if we find that cuda is initialized, we inspect the current function call stack, and trace back the stack until we reach the `__main__` module, check the current line to see if we are under `if __name__ == "__main__"`, if yes, switch the multiprocessing method from `fork` to `spawn`.

cc @russellb 

### Alternatives

_No response_

### Additional context

_No response_

[... truncated for brevity ...]

---

## Issue #N/A: [Tracking Issue]: Multi-modal model requests

**Link**: https://github.com/vllm-project/vllm/issues/14876
**State**: closed
**Created**: 2025-03-16T02:14:05+00:00
**Closed**: 2025-03-16T13:01:04+00:00
**Comments**: 0
**Labels**: new-model, multi-modality

### Description

Moved to https://github.com/orgs/vllm-project/projects/10

---

## Issue #N/A: [RFC]: Usage Data Enhancement for v0.5.*

**Link**: https://github.com/vllm-project/vllm/issues/5520
**State**: closed
**Created**: 2024-06-14T01:26:36+00:00
**Closed**: 2024-11-27T02:06:47+00:00
**Comments**: 5
**Labels**: RFC, stale

### Description

### Motivation.

vLLM currently has a usage reporting feature https://docs.vllm.ai/en/stable/serving/usage_stats.html to inform us what features can be safely deprecated or what hardware to improve performance on.

After v0.5.0, vLLM has various features that's being tested (chunked prefill, prefix caching, spec decode, fp8, and VLM), we would like to start gathering statistics on the usage of these features with different hardware and model types so we know what we are tested on. 

### Proposed Change.

Add the following data to `usage_lib`
* `--enable-chunked-prefill`
* `--enable-prefix-cache`
* `speculative_model` (need model architecture/size or [ngram])

Another missing value from previous data is the size of the model, so we find it difficult to compare llama3 8b vs 70b. This might require some creative way to find the size of the model without capturing too much information. 

Any other suggestion welcomed.

### Feedback Period.

_No response_

### CC List.

_No respons

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Build error, nvcc error   : 'ptxas' died due to signal 11 (Invalid memory reference)

**Link**: https://github.com/vllm-project/vllm/issues/15452
**State**: closed
**Created**: 2025-03-25T08:58:06+00:00
**Closed**: 2025-03-25T09:40:51+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

L20, CUDA 12.6, PyTorch 2.6.0

### 🐛 Describe the bug

```bash
FAILED: vllm-flash-attn/CMakeFiles/_vllm_fa3_C.dir/hopper/instantiations/flash_fwd_hdimall_fp16_paged_split_sm90.cu.o
/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -DFLASHATTENTION_DISABLE_BACKWARD -DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_PYBIND -DFLASHATTENTION_DISABLE_UNEVEN_K -DFLASHATTENTION_VARLEN_ONLY -DPy_LIMITED_API=3 -DTORCH_EXTENSION_NAME=_vllm_fa3_C -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -D_vllm_fa3_C_EXPORTS -I/workspace/dev/vipshop/vllm/.deps/vllm-flash-attn-src/csrc -I/workspace/dev/vipshop/vllm/.deps/vllm-flash-attn-src/hopper -I/workspace/dev/vipshop/vllm/.deps/vllm-flash-attn-src/csrc/common -I/workspace/dev/vipshop/vllm/.deps/vllm-flash-attn-src/csrc/cutlass/include -isystem /usr/include/python3.12 -isystem /usr/local/lib/python3.12/dist-packages/torch/include -isystem /usr/local/lib/python3.12/dist-packages/t

[... truncated for brevity ...]

---

## Issue #N/A: Check whether the input request is too long

**Link**: https://github.com/vllm-project/vllm/issues/113
**State**: closed
**Created**: 2023-05-20T23:02:26+00:00
**Closed**: 2024-04-10T08:47:01+00:00
**Comments**: 4
**Labels**: bug

### Description

No description provided.

---

## Issue #N/A: [Bug]: vision chat completion output with odd Instruction/Output prompting.

**Link**: https://github.com/vllm-project/vllm/issues/5693
**State**: closed
**Created**: 2024-06-19T21:12:05+00:00
**Closed**: 2024-06-25T06:57:56+00:00
**Comments**: 23
**Labels**: bug

### Description

### Your current environment

```text
git clone https://github.com/vllm-project/vllm.git
cd ~/vllm
conda create -n vllm -y
conda activate vllm
conda install python=3.10 -y
pip install -e .
pip install hf_transfer
pip install torchvision
```
latest main afed90a0344b1b0ce6aae46efc630adb489ec769

run:
```
export NCCL_IGNORE_DISABLED_P2P=1
export CUDA_VISIBLE_DEVICES=5
python -m vllm.entrypoints.openai.api_server --port=5063 \
      --host=0.0.0.0 --model microsoft/Phi-3-vision-128k-instruct \
      --tensor-parallel-size=1 --seed 1234 \
      --max-num-batched-tokens=8192        \
      --trust-remote-code \
      --tensor-parallel-size=1 \
      --max-num-batched-tokens=131072 --max-log-len=100 \
      --image-input-type=pixel_values \
      --image-token-id=32044 \
      --image-input-shape="1,3,1008,1344" \
      --image-feature-size=1921 \
      --download-dir=$HOME/.cache/huggingface/hub &> vllm_phi3_vision.log &
```

### 🐛 Describe the bug

```
from open

[... truncated for brevity ...]

---

## Issue #N/A: Slow build speed with punica after #1804

**Link**: https://github.com/vllm-project/vllm/issues/2571
**State**: closed
**Created**: 2024-01-24T00:44:29+00:00
**Closed**: 2024-01-29T13:29:22+00:00
**Comments**: 3

### Description

After #1804, the build speed of vLLM becomes significantly slower. Most of the time was spent on `/home/zhuohan/vllm/vllm/csrc/punica/bgmv/bgmv_all.cu`. Should we turn off punica by default?

cc @Yard1 

---

## Issue #N/A: [Bug][V1]: TP is broken when torch compile cache is used

**Link**: https://github.com/vllm-project/vllm/issues/13435
**State**: closed
**Created**: 2025-02-17T22:58:05+00:00
**Closed**: 2025-02-18T04:33:47+00:00
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


Got the error message when using tp_size=4:
```
(VllmWorker rank=2 pid=2307184) ERROR 02-17 14:48:01 multiproc_executor.py:374] ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
```
**Importantly, the bug doesn't happen when the torch.compile cache is not used.**

The error raises at the first torch.compile-generated op for the embedding layer:
```python
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s0, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [ge, lt, and_, ge_1, lt_1, and__1, or_, masked_fill_, mul, mul_1, add, sub, mul_2, embedding], Original ATen: [aten.ge, aten.lt, aten.bitwise_and, aten.bitwise_or, aten.masked_fill, aten.mul, aten.add, aten.sub, aten.embedding]
   

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: why vllm-0.6.1.post2 faster than latest vllm=0.6.6.post1?

**Link**: https://github.com/vllm-project/vllm/issues/12274
**State**: closed
**Created**: 2025-01-21T18:36:39+00:00
**Closed**: 2025-05-23T02:10:28+00:00
**Comments**: 2
**Labels**: performance, stale

### Description

### Your current environment

old: vllm-0.6.1.post2 
new:  vllm=0.6.6.post1


### Model Input Dumps

_No response_

### 🐛 Describe the bug

I used the previous version of vllm-0.6.1.post2 and I did benchmark to get the max TTFT and then I upgrade to the latest vllm=0.6.6.post1, but when I did the benchmark again I see a huge difference in the performance regarding the TTFT!

benchmarking llama3.1-70b-awq model, with 20 1k request on 4gpus, the max TTFT was 10 seconds for the previous vllm but with the latest vllm it increased to be 37 seconds!!

Any thoughts why this huge difference in TTFT here? Do I miss any configs or args to be set to make it faster?

Thanks.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: Don't placement_group support NPU?

**Link**: https://github.com/vllm-project/vllm/issues/18197
**State**: closed
**Created**: 2025-05-15T09:33:50+00:00
**Closed**: 2025-05-24T02:34:06+00:00
**Comments**: 0
**Labels**: usage

### Description

https://github.com/vllm-project/vllm-ascend/issues/870


---

