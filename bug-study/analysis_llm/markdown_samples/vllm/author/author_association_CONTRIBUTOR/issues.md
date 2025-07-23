# author_association_CONTRIBUTOR - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 27

### Label Distribution

- bug: 14 issues
- stale: 10 issues
- feature request: 6 issues
- performance: 1 issues
- usage: 1 issues
- good first issue: 1 issues
- installation: 1 issues

---

## Issue #N/A: install from source failed

**Link**: https://github.com/vllm-project/vllm/issues/3180
**State**: closed
**Created**: 2024-03-04T16:21:49+00:00
**Closed**: 2024-03-05T07:11:15+00:00
**Comments**: 2

### Description

I encountered a problem when trying to compile vllm using 'pip install -e .', but I succeeded using 'pip install vllm'
Here is my env configuration steps

```
conda create --name vllm python=3.9
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements.txt
pip install -e .
```

Env:
cuda 12.1
python 3.9.18
numpy 1.26.4
ninja 1.11.1.1

Error:

```
  Building editable for vllm (pyproject.toml) ... error
  error: subprocess-exited-with-error


  × Building editable for vllm (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [221 lines of output]
      /tmp/pip-build-env-dy69uemu/overlay/lib/python3.9/site-packages/torch/nn/modules/transformer.py:20: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:84.)
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
      running editable_wheel
      creating /tmp/pip-wheel-nzc13jkb/

[... truncated for brevity ...]

---

## Issue #N/A: [Enhancement]:  add concurrency in benchmark result table

**Link**: https://github.com/vllm-project/vllm/issues/21094
**State**: open
**Created**: 2025-07-17T05:32:47+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

vLLM v0.9.2rc2

### 🐛 Describe the bug

when using `vllm/benchmarks/benchmark_serving.py`

the  `--max-concurrency X` only shows in normal stdout logging and saving result json,

 but is missing in **formatted result table** (`============ Serving Benchmark Result ============
`)

the concurrency is important factor impact the performance result , sometimes  even than  `Successful requests`.

So I think it's necessary to put this info into  formatted result table output.




current output table:

```Namespace(backend='openai-chat', base_url='https://api.groq.com/openai', host='127.0.0.1', port=8000, endpoint='/v1/chat/completions', dataset_name='sharegpt', dataset_path='/mnt/models/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json', max_concurrency=3, model='moonshotai/Kimi-K2-Instruct', tokenizer=None, use_beam_search=False, num_prompts=10, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=True, disable_tqdm=Fa

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

## Issue #N/A: [Feature]: Multi-Proposers support for speculative decoding.

**Link**: https://github.com/vllm-project/vllm/issues/6300
**State**: closed
**Created**: 2024-07-10T09:36:55+00:00
**Closed**: 2024-11-25T02:04:43+00:00
**Comments**: 7
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

Speculative decoding has demonstrated significant potential in efficiently generating proposals and utilizing idle computing power to expedite the auto-regression decoding process, particularly under lightweight workloads. Thanks to the remarkable work by @cadedaniel, we have verified the latency benefits brought by speculative decoding on the latest version of vllm.

We have observed the following points that we believe could further enhance the utility of speculative decoding:

* **Ngram Proposer:** While the 'Ngram' proposer can offer a 2x to 3x performance improvement in Retrieval-Augmented Generation (RAG) scenarios, its performance diminishes when the RAG module retrieves no relevant data for a query.

* **Draft-Model-Based Proposers:** In contrast, draft-model-based proposers have exhibited higher acceptance rates when the RAG module retrieves no relevant data or faces a more creative task. Yet the performance of this type of implem

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Chunk Prefill feature fails for ppc64le (IBM POWER)

**Link**: https://github.com/vllm-project/vllm/issues/13387
**State**: closed
**Created**: 2025-02-17T08:54:37+00:00
**Closed**: 2025-06-25T02:16:37+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.7.0a0+gitd0f5df8
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Red Hat Enterprise Linux 9.5 (Plow) (ppc64le)
GCC version: (GCC) 13.3.1 20240611 (Red Hat 13.3.1-2)
Clang version: 18.1.8 (Red Hat, Inc. 18.1.8-3.el9)
CMake version: version 3.31.2
Libc version: glibc-2.34

Python version: 3.11.11 | packaged by conda-forge | (main, Dec  5 2024, 14:07:52) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.14.0-503.23.1.el9_5.ppc64le-ppc64le-with-glibc2.34
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False

CPU:
Architecture:                         ppc64le
Byte Order:          

[... truncated for brevity ...]

---

## Issue #N/A: OpenBuddy support

**Link**: https://github.com/vllm-project/vllm/issues/277
**State**: closed
**Created**: 2023-06-27T10:42:21+00:00
**Closed**: 2023-06-27T15:27:20+00:00
**Comments**: 1

### Description

I'd very much like to see support for [OpenBuddy](https://github.com/OpenBuddy/OpenBuddy) in vllm.

---

## Issue #N/A: When is the CPU KV cache used and swapping?

**Link**: https://github.com/vllm-project/vllm/issues/2853
**State**: closed
**Created**: 2024-02-13T18:58:19+00:00
**Closed**: 2024-08-27T09:52:39+00:00
**Comments**: 3

### Description

Hi authors,

In your implementation, the GPU memory is leveraged to store the KV cache. However, it appears that when the GPU memory reaches its capacity, there isn't a mechanism in place to offload or swap this data to the CPU memory. 

1. Could you please clarify under what conditions the CPU KV cache comes into play?
2. Could you please tell me how to invoke the CPU KV cache (or API) if I want to do swapping?


![Screenshot 2024-02-01 at 2 41 42 PM](https://github.com/vllm-project/vllm/assets/47625290/38ee7a92-486d-4e2a-a7f1-3c8af2424cd4)
![Screenshot 2024-02-13 at 1 53 57 PM](https://github.com/vllm-project/vllm/assets/47625290/618227a2-c599-4028-b47d-ae03e0762016)


---

## Issue #N/A: [Feature]: AssertionError: MolmoForCausalLM does not support LoRA yet.

**Link**: https://github.com/vllm-project/vllm/issues/11431
**State**: closed
**Created**: 2024-12-23T10:38:18+00:00
**Closed**: 2024-12-31T01:33:08+00:00
**Comments**: 3
**Labels**: feature request

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.12 (main, Sep 11 2024, 15:47:36) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-122-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.5.82
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA RTX A6000
GPU 1: NVIDIA RTX A6000

Nvidia driver version: 555.42.02
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address 

[... truncated for brevity ...]

---

## Issue #N/A: Support for Qwen 1.5 on vLLM

**Link**: https://github.com/vllm-project/vllm/issues/2814
**State**: closed
**Created**: 2024-02-08T14:31:13+00:00
**Closed**: 2024-02-23T09:26:41+00:00
**Comments**: 5

### Description

Currently, this gives this error:
```
2024-02-08T06:28:20.354023087-08:00 OSError: Can't load the configuration of 'Qwen/Qwen1.5-72B-Chat-AWQ'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'Qwen/Qwen1.5-72B-Chat-AWQ' is the correct path to a directory containing a config.json file
```

Reference implementation: [here](https://runpod.io/gsc?template=ju7oo9mf5w&ref=jmfkcdio)

---

## Issue #N/A: [Bug]: With Pytorch 2.7 in case of POWER getting, NotImplementedError: Could not run '_C_cache_ops::reshape_and_cache' with arguments from the 'CPU' backend

**Link**: https://github.com/vllm-project/vllm/issues/17960
**State**: closed
**Created**: 2025-05-11T12:59:05+00:00
**Closed**: 2025-05-19T06:20:16+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
INFO 05-11 07:55:22 [__init__.py:248] Automatically detected platform cpu.
Collecting environment information...
PyTorch version: 2.7.0
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Red Hat Enterprise Linux 9.5 (Plow) (ppc64le)
GCC version: (GCC) 11.5.0 20240719 (Red Hat 11.5.0-5)
Clang version: 19.1.7 (CentOS 19.1.7-1.el9)
CMake version: version 3.31.6
Libc version: glibc-2.34

Python version: 3.12.9 (main, Feb  4 2025, 00:00:00) [GCC 11.5.0 20240719 (Red Hat 11.5.0-4)] (64-bit runtime)
Python platform: Linux-5.14.0-547.el9.ppc64le-ppc64le-with-glibc2.34
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False

CPU:
Architect

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: lazy import for VLM

**Link**: https://github.com/vllm-project/vllm/issues/6187
**State**: closed
**Created**: 2024-07-07T11:50:48+00:00
**Closed**: 2024-07-08T05:20:06+00:00
**Comments**: 2
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

I used [vLLM 0.5.0.post1](https://github.com/vllm-project/vllm/releases/tag/v0.5.0.post1) for `Mixtral-8x7B-Instruct-v0.1` inference
```bash
python3 -m vllm.entrypoints.openai.api_server --model /workdir/Mixtral-8x7B-Instruct-v0.1 --tensor-parallel-size 2
```
and get the error
```
Traceback (most recent call last):
  File "/usr/local/lib/python3.9/site-packages/transformers/utils/import_utils.py", line 1560, in _get_module
    return importlib.import_module("." + module_name, self.__name__)
  File "/usr/local/lib/python3.9/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1030, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1007, in _find_and_load
  File "<frozen importlib._bootstrap>", line 986, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 680, in _load_unlocked
  File "<frozen im

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: wrong output on L20 using fp8

**Link**: https://github.com/vllm-project/vllm/issues/19779
**State**: open
**Created**: 2025-06-17T23:08:07+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
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
PyTorch version              : 2.6.0+cu124
Is debug build               : False
CUDA used to build PyTorch   : 12.4
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0] (64-bit runtime)
Python platform              : Linux-5.4.210.bsk.6-amd64-x86_64-with-glibc2.35

==============================
    

[... truncated for brevity ...]

---

## Issue #N/A: Question: Would a PR integrating ExLlamaV2 kernels with AWQ be accepted?

**Link**: https://github.com/vllm-project/vllm/issues/2645
**State**: closed
**Created**: 2024-01-29T09:42:35+00:00
**Closed**: 2025-01-19T02:02:12+00:00
**Comments**: 7
**Labels**: performance, feature request, stale

### Description

Recently, ExLlamaV2 kernels were introduced into AutoAWQ. We can instantly map the AWQ packed weights to be compatible with ExLlama, and it runs decoding about 20% faster.

# Performance

Note that the gap in prefilling has recently been closed, so the main benefit would be during decoding.

### GEMM (AWQ kernel)

| Batch Size | Prefill Length | Decode Length | Prefill tokens/s | Decode tokens/s | Memory (VRAM) |
| -- | -- | -- | -- | -- | -- |
| 1 | 64 | 64 | 316\.842 | 156\.038 | 4\.78 GB (20.20%) |
| 1 | 128 | 128 | 4898\.86 | 154\.977 | 4\.79 GB (20.27%) |
| 1 | 256 | 256 | 5366\.24 | 151\.31 | 4\.81 GB (20.35%) |
| 1 | 512 | 512 | 5239\.46 | 144\.517 | 4\.85 GB (20.51%) |
| 1 | 1024 | 1024 | 4573\.25 | 132\.849 | 4\.93 GB (20.83%) |
| 1 | 2048 | 2048 | 3859\.42 | 114\.249 | 5\.55 GB (23.48%) |
| 8 | 64 | 64 | 1733\.1 | 1176\.07 | 4\.83 GB (20.42%) |
| 8 | 128 | 128 | 5359\.34 | 1167\.19 | 4\.90 GB (20.72%) |
| 8 | 256 | 256 | 5145\.94 | 1130\.84 | 5\.03 GB (21.26%) |
| 8 | 512 | 5

[... truncated for brevity ...]

---

## Issue #N/A: Attention sliding window

**Link**: https://github.com/vllm-project/vllm/issues/3385
**State**: closed
**Created**: 2024-03-13T19:53:48+00:00
**Closed**: 2025-03-28T02:05:33+00:00
**Comments**: 13
**Labels**: stale

### Description

In Hugging Face "eager" Mistral implementation, a sliding window of size 2048 will mask 2049 tokens. This is also true for flash attention. In the current vLLM implementation a window of 2048 will mask 2048 tokens:

```
import torch
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

attn_bias = BlockDiagonalCausalMask.from_seqlens([4096])
attn_bias = attn_bias.make_local_attention(2048)
mask = attn_bias._create_block_mask([4096, 4096])
print(torch.sum(mask == 0, dim=1))
```
**Output: tensor([   1,    2,    3,  ..., 2048, 2048, 2048])**


The output should be: **tensor([   1,    2,    3,  ..., 2049, 2049, 2049])**

Context: https://github.com/huggingface/transformers/issues/29623 

---

## Issue #N/A: [Bug]: Failing to find LoRA adapter for MultiLoRA Inference

**Link**: https://github.com/vllm-project/vllm/issues/4520
**State**: closed
**Created**: 2024-05-01T08:49:00+00:00
**Closed**: 2024-11-28T02:05:44+00:00
**Comments**: 4
**Labels**: bug, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### 🐛 Describe the bug

I'm running the latest docker image and an openai style endpoint.

My command is:
```
--model NousResearch/Meta-Llama-3-8B-Instruct --max-model-len 8192 --port 8000 --enable-lora --lora-modules forced-french=Trelis/Meta-Llama-3-8B-Instruct-forced-french-adapters --max-loras 1 --max-lora-rank 8
```
I'm hitting the endpoint (on runpod) with:
```
curl https://y55xy7ozoxrn15-8000.proxy.runpod.net/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "forced-french",
        "prompt": "Why did the chicken cross the road?",
        "max_tokens": 50,
        "temperature": 0
    }'
```
The error is:
```
terminal: Internal Server Error

logs:

2024-05-01T08:45:12.669025667Z ERROR 05-01 08:45:12 async_llm_engine.py:43]     return func(*args, **kwargs)
2024-05-01T08:45:12.669029441Z ERROR 05-01 08:45:12 async_llm_engine.py:43]   File

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: model_executor/test_model_load_with_params.py  fails with AttributeError

**Link**: https://github.com/vllm-project/vllm/issues/18757
**State**: closed
**Created**: 2025-05-27T10:01:18+00:00
**Closed**: 2025-05-28T05:42:56+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

Issue encountered on main branch tests.

### 🐛 Describe the bug

Test failing with below traceback:

```
vllm_runner = <class 'tests.conftest.VllmRunner'>

    @pytest.mark.skipif(current_platform.is_rocm(),
                        reason="Xformers backend is not supported on ROCm.")
    def test_model_loading_with_params(vllm_runner):
        """
        Test parameter weight loading with tp>1.
        """
        with vllm_runner(model_name=MODEL_NAME,
                         revision=REVISION,
                         dtype="float16",
                         max_model_len=MAX_MODEL_LEN) as vllm_model:
            output = vllm_model.encode("Write a short story about a robot that"
                                       " dreams for the first time.\n")
    
            model_config = vllm_model.model.llm_engine.model_config
            model_tokenizer = vllm_model.model.llm_engine.tokenizer
    
            # asserts on the bert model config file
      

[... truncated for brevity ...]

---

## Issue #N/A: How to install from source with CUDA 11.8 instead of 12.1?

**Link**: https://github.com/vllm-project/vllm/issues/2072
**State**: closed
**Created**: 2023-12-13T01:49:45+00:00
**Closed**: 2024-03-28T12:02:21+00:00
**Comments**: 4

### Description

No description provided.

---

## Issue #N/A: [Bug]: glm.py rotary_dim bug

**Link**: https://github.com/vllm-project/vllm/issues/16904
**State**: closed
**Created**: 2025-04-21T06:29:55+00:00
**Closed**: 2025-04-21T14:26:35+00:00
**Comments**: 2
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

class GlmForCausalLM(LlamaForCausalLM, SupportsV0Only):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Hack Llama model to fit HF format GLM implementation
        # Attention difference between GLM and Llama:
        # 1. Half partial rotary_dim and no Neox style.
        # 2. There is no bias for o_proj in attention
        for layer in self.model.layers:
            if not isinstance(layer, PPMissingLayer):
                print(layer.self_attn.rotary_emb.rotary_dim)
                layer.self_attn.rotary_emb.rotary_dim //= 2
                layer.self_attn.rotary_emb.is_neox_style = False
                layer.self_attn.o_proj.bias = None
                layer.self_attn.o_proj.skip_bias_add = Tru

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: guide decoding lead to an incorrect function call arguments

**Link**: https://github.com/vllm-project/vllm/issues/7774
**State**: closed
**Created**: 2024-08-22T07:17:46+00:00
**Closed**: 2024-12-22T02:04:22+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
WARNING 08-22 15:09:07 _custom_ops.py:14] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
[DLI_CUDA] cudaDeviceGetStreamPriorityRange is unsupported, and return cudaSuccess.
PyTorch version: 2.2.1
Is debug build: False
CUDA used to build PyTorch: 11.7
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: 15.0.6
CMake version: version 3.27.0
Libc version: glibc-2.31

Python version: 3.10.8 (main, Nov 24 2022, 14:13:03) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-105-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: Could not collect
Nvidia driver version: Could not collect
cuDNN version: C

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Distinguish LoRA Model Metrics from Base Model Metrics in Reporting

**Link**: https://github.com/vllm-project/vllm/issues/11091
**State**: closed
**Created**: 2024-12-11T09:17:15+00:00
**Closed**: 2025-04-11T02:06:13+00:00
**Comments**: 4
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

When submitting requests to a LoRA model and subsequently checking the associated metrics, I've noticed that all metrics are aggregated under the base model's metrics. This means that requests made to the LoRA model are being counted as requests to the base model. Given that LoRA models logically represent a distinct model layer on top of the base, it is crucial for accurate monitoring and analysis that we separate these metrics.

part of https://github.com/vllm-project/vllm/issues/6275

```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sql-lora",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }' 
```


![image](https://github.com/user-attachments/assets/cf7706d3-e4f2-4f92-985b-517a90b1e3c6)


### Expected Behavior

Metrics for LoRA models should be distinctly reported, separate from the base model metrics, to

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Running vLLM with B200 Blackwell

**Link**: https://github.com/vllm-project/vllm/issues/17901
**State**: closed
**Created**: 2025-05-09T12:43:42+00:00
**Closed**: 2025-05-29T21:26:28+00:00
**Comments**: 3
**Labels**: usage

### Description

### Your current environment

I have been trying the latest image but it seems not to use cuda 12.8, which is needed. Is there an image that will work with vllm and B200?

### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: v1 engine with full cuda graph option outputs garbage

**Link**: https://github.com/vllm-project/vllm/issues/18533
**State**: closed
**Created**: 2025-05-22T07:19:36+00:00
**Closed**: 2025-06-04T08:10:24+00:00
**Comments**: 4
**Labels**: bug

### Description

### Your current environment

x

### 🐛 Describe the bug

refers to https://github.com/vllm-project/vllm/issues/18520

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: AsyncLLMEngine cannot stop iteration when generation completes

**Link**: https://github.com/vllm-project/vllm/issues/3024
**State**: closed
**Created**: 2024-02-24T17:23:31+00:00
**Closed**: 2024-02-27T09:08:31+00:00
**Comments**: 7

### Description

@WoosukKwon @simon-mo 

### Environment

- torch 2.1.2
- vllm 0.3.2

### Reproduce

```python
import asyncio
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs("01-ai/Yi-6B-Chat"))
param = SamplingParams(max_tokens=50, stop_token_ids=[7])
generator = engine.generate("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", param, "req_test")

async def test():
    answer = None
    async for result in generator:
        print(result.finished, "-", result.outputs[0].text)
        answer = result.outputs[0].text
    print("Answer:", answer)

asyncio.get_event_loop().run_until_complete(test())
```

### Outputs

```
False - Hello
False - Hello!
False - Hello! How
False - Hello! How can
False - Hello! How can I
False - Hello! How can I assist
False - Hello! How can I assist you
False - Hello! How can I assist you today
False - Hello! How can I assist you today?
INFO 02-25 01:20:26

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: qwen2-72b-instruct model  with RuntimeError: CUDA error: an illegal memory access was encountered 

**Link**: https://github.com/vllm-project/vllm/issues/6776
**State**: closed
**Created**: 2024-07-25T07:28:58+00:00
**Closed**: 2025-05-17T02:09:45+00:00
**Comments**: 6
**Labels**: bug, stale

### Description

### Your current environment

```text
PyTorch version: 2.3.0a0+ebedce2
Is debug build: False
CUDA used to build PyTorch: 12.3
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.0
Libc version: glibc-2.35

Python version: 3.10.12 (main, Mar 22 2024, 16:50:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-4.19.91-014-kangaroo.2.10.13.5c249cdaf.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.3.107
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB
GPU 2: NVIDIA A100-SXM4-80GB
GPU 3: NVIDIA A100-SXM4-80GB

Nvidia driver version: 535.54.03
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.0.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.0.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.0.0
/usr/lib/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: : CPU silently doesn't support multi-step (--num-scheduler-steps)

**Link**: https://github.com/vllm-project/vllm/issues/8477
**State**: closed
**Created**: 2024-09-13T19:55:13+00:00
**Closed**: 2025-01-13T02:03:03+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
INFO 09-13 19:13:45 importing.py:10] Triton not installed; certain GPU-related functions will not be available.
PyTorch version: 2.4.0+cpu
Is debug build: False
CUDA used to build PyTorch: Could not collect
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
Clang version: Could not collect
CMake version: version 3.30.2
Libc version: glibc-2.35

Python version: 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-4.18.0-372.46.1.el8_6.x86_64-x86_64-with-glibc2.35
Is CUDA available: False
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: 
GPU 0: NVIDIA A100-SXM4-80GB
  MIG 3g.40gb     Device  0:

Nvidia driver version: 535.104.05
cuDNN version: Could not collect
HIP 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: torch.ops._C.cutlass_scaled_mm RuntimeError: Error Internal while use L20 PP=3 + TP=8 for R1

**Link**: https://github.com/vllm-project/vllm/issues/14601
**State**: closed
**Created**: 2025-03-11T07:34:43+00:00
**Closed**: 2025-03-11T12:16:39+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
INFO 03-11 15:29:14 [__init__.py:256] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 14.0.0-1ubuntu1.1
CMake version: version 3.29.2
Libc version: glibc-2.35

Python version: 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-4.18.0-348.7.1.el8_5.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.6.85
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

Nvidia driver version: 550.5

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: support `stream_options` option

**Link**: https://github.com/vllm-project/vllm/issues/4967
**State**: closed
**Created**: 2024-05-22T02:44:58+00:00
**Closed**: 2024-06-07T03:29:26+00:00
**Comments**: 1
**Labels**: good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

According to openAI doc: https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options. The API provide the stream_options which can get token usage info for stream request. 

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [Installation]: Fail to build vLLM from source on CUDA 12.6

**Link**: https://github.com/vllm-project/vllm/issues/15435
**State**: open
**Created**: 2025-03-25T03:51:53+00:00
**Comments**: 11
**Labels**: installation

### Description

### Your current environment

```text
INFO 03-24 20:48:52 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: CentOS Stream 9 (x86_64)
GCC version: (GCC) 11.5.0 20240719 (Red Hat 11.5.0-5)
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.34

Python version: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.4.3-0_fbk14_hardened_2601_gcd42476b84e9-x86_64-with-glibc2.34
Is CUDA available: True
CUDA runtime version: 12.6.85
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA H100
GPU 1: NVIDIA H100
GPU 2: NVIDIA H100
GPU 3: NVIDIA H100
GPU 4: NVIDIA H100
GPU 5: NVIDIA H100
GPU 6: NVIDIA H100
GPU 7: NVIDIA H100

Nvidia driver version: 550.90.07
cuDNN version: Probably one of the following:
/usr/lib6

[... truncated for brevity ...]

---

## Issue #N/A: --tensor-parallel-size 2 fails to load on GCP

**Link**: https://github.com/vllm-project/vllm/issues/2906
**State**: closed
**Created**: 2024-02-18T06:49:47+00:00
**Closed**: 2024-11-30T02:02:35+00:00
**Comments**: 10
**Labels**: stale

### Description

Hi,
I am trying to set up vLLM Mixtral 8x7b on GCP. I have a VM with two A100 80GBs, and am using the following setup:

docker image: vllm/vllm-openai:v0.3.0
Model: mistralai/Mixtral-8x7B-Instruct-v0.1

Command I use inside the vm:

python3 -m vllm.entrypoints.openai.api_server --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tensor-parallel-size 2 --port 8888

Output (after a while):

```
File "/usr/local/lib/python3.10/dist-packages/torch/nn/functional.py", line 1858, in softmax
    ret = input.softmax(dim, dtype=dtype)
RuntimeError: CUDA error: invalid device function
```

nvidia-smi output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usa

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Incorrect Example for the Inference with Prefix 

**Link**: https://github.com/vllm-project/vllm/issues/5177
**State**: closed
**Created**: 2024-06-01T13:06:28+00:00
**Closed**: 2024-06-01T22:53:53+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

```text
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.3
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.2.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100 80GB PCIe
GPU 1: NVIDIA A100 80GB PCIe
GPU 2: NVIDIA A100 80GB PCIe
GPU 3: NVIDIA A100 80GB PCIe

Nvidia driver version: 535.161.07
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address siz

[... truncated for brevity ...]

---

