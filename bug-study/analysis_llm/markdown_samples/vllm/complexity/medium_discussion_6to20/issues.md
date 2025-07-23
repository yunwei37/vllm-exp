# medium_discussion_6to20 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 5
- Closed Issues: 25

### Label Distribution

- bug: 15 issues
- stale: 8 issues
- feature request: 5 issues
- usage: 2 issues
- performance: 1 issues
- torch.compile: 1 issues
- v1: 1 issues
- installation: 1 issues
- misc: 1 issues
- help wanted: 1 issues

---

## Issue #N/A: [Bug]: Can't run vllm entrypoint after a fresh install because of a bug in huggingface-hub 

**Link**: https://github.com/vllm-project/vllm/issues/3796
**State**: closed
**Created**: 2024-04-02T13:19:07+00:00
**Closed**: 2024-04-02T21:47:34+00:00
**Comments**: 9
**Labels**: bug

### Description

### Your current environment

```text
2024-04-02 13:00:28 (5.94 MB/s) - ‘collect_env.py’ saved [24853/24853]

Collecting environment information...
/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
PyTorch version: 2.1.2+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.0
Libc version: glibc-2.35

Python version: 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 11.8.89
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 3090
Nvidia driver version: 550.54.14
c

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Qwen2.5-Math-7B-Instruct vllm output garbled code, but the probability of huggingface outputing garbled code is lower.

**Link**: https://github.com/vllm-project/vllm/issues/9183
**State**: closed
**Created**: 2024-10-09T07:07:21+00:00
**Closed**: 2024-12-27T03:50:22+00:00
**Comments**: 19
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
2024-10-09 14:34:40 (641 KB/s) - ‘collect_env.py’ saved [25599/25599]

Collecting environment information...
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.4
Libc version: glibc-2.35

Python version: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.4.0-155-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 11.8.89
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A800-SXM4-80GB
GPU 1: NVIDIA A800-SXM4-80GB
GPU 2: NVIDIA A800-SXM4-80GB
GPU 3: NVIDIA A800-SXM4-80GB

Nvidia driver version: 535.54.03
cuDNN version: Probably one of the following:
/u

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: ValueError: Attempted to assign 25 + 25 + 25 = 75 multimodal tokens to 147 placeholders

**Link**: https://github.com/vllm-project/vllm/issues/18572
**State**: open
**Created**: 2025-05-22T22:16:11+00:00
**Comments**: 6
**Labels**: bug

### Description

### Your current environment

vllm==0.7.3   transformers==4.49.0



### 🐛 Describe the bug

```import os 
os.environ["HF_HOME"] = "/gz-data/.cache/huggingface"
import torch
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    TrainerCallback, 
    TrainerControl, 
    TrainerState,
    BitsAndBytesConfig
)
import json
import mlflow
import os
from transformers.integrations import HfDeepSpeedConfig
import shutil
from Levenshtein import ratio as levenshtein_ratio
from vllm_grpo_trainer import Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig
from PIL import Image

def process_func_for_grpo(example):
    conversation = example["conversations"]
    input_msg = conversation[0]["value"]
    output_msg = conversation[1]["value"]

    file

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: reproducing vLLM performance benchmark

**Link**: https://github.com/vllm-project/vllm/issues/8176
**State**: closed
**Created**: 2024-09-05T04:47:18+00:00
**Closed**: 2024-12-23T18:48:16+00:00
**Comments**: 8
**Labels**: performance, stale

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

_No response_

### Misc discussion on performance

To reproduce vLLM's performance benchmark, please launch a shell in the following docker images:

- SGlang: `lmsysorg/sglang:v0.3.0-cu124`
- lmdeploy: `openmmlab/lmdeploy:v0.6.0a0-cu12`
- TensorRT-LLM: `nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3`
- vLLM: `vllm/vllm-openai:v0.6.0`

And then run the following bash script (don't forget to replace <your HF TOKEN> with your huggingface token that has Llama-3 model access):
```bash
export HF_TOKEN=<your HF TOKEN>
apt update
apt install -y wget unzip 
# download benchmarking code
wget -O benchmarking_code.zip https://buildkite.com/organizations/vllm/pipelines/performance-benchmark/builds/8532/jobs/0191bbbf-c603-4c15-9f5d-e0b2933ba097/artifacts/0191bd2a-d6cd-4f6d-b618-a7aa2c39456c
unzip benchmarking_code.zip
# remove previous results
rm -r ./benchmarks/results
VLLM_SOUR

[... truncated for brevity ...]

---

## Issue #N/A: `8-bit quantization` support

**Link**: https://github.com/vllm-project/vllm/issues/214
**State**: closed
**Created**: 2023-06-22T23:02:03+00:00
**Closed**: 2024-04-18T13:52:08+00:00
**Comments**: 14
**Labels**: feature request

### Description

As far as I know `vllm` and `ray` doesn't support `8-bit quantization` as of now. I think it's the most viable quantization technique out there and should be implemented for faster inference and reduced memory usage. 

---

## Issue #N/A: [Bug]: vllm container does not set LD_LIBRARY_PATH correctly

**Link**: https://github.com/vllm-project/vllm/issues/12559
**State**: closed
**Created**: 2025-01-29T17:30:27+00:00
**Closed**: 2025-06-01T02:14:31+00:00
**Comments**: 8
**Labels**: bug, stale

### Description

### Your current environment

vllm container 0.7.0

### Model Input Dumps

_No response_

### 🐛 Describe the bug

mycontainer# env|grep LD_ 
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

But...
/usr/local/nvidia does not exist in the container

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.


---

## Issue #N/A: [Bug]: vllm stall on llama3-70b warmup with 0.4.1

**Link**: https://github.com/vllm-project/vllm/issues/4277
**State**: closed
**Created**: 2024-04-22T22:41:13+00:00
**Closed**: 2024-06-13T09:02:35+00:00
**Comments**: 19
**Labels**: bug

### Description

### Your current environment

I'm running a minimally modified image of `nvcr.io/nvidia/pytorch:23.10-py3` with Python 3.11.

```
PyTorch version: 2.2.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.2
Libc version: glibc-2.35

Python version: 3.11.5 (main, Aug 26 2023, 07:22:50) [Clang 16.0.3 ] (64-bit runtime)
Python platform: Linux-4.4.0-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.2.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB

Nvidia driver version: 535.129.03
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.5
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.5
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.5
/usr

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: DeepSeek-R1 KeyError: 'layers.61.mlp.experts.w2_weight'

**Link**: https://github.com/vllm-project/vllm/issues/16450
**State**: closed
**Created**: 2025-04-11T03:33:48+00:00
**Closed**: 2025-04-11T06:44:24+00:00
**Comments**: 6
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

- cmd

```bash
export VLLM_USE_V1=0 && export VLLM_PP_LAYER_PARTITION="22,20,19"
nohup python3 -m vllm.entrypoints.openai.api_server \
        --model=/workspace/dev/hf_models/DeepSeek-R1 \
        --dtype=auto \
        --block-size 32 \
        --tokenizer-mode=slow \
        --max-model-len 32768 \
        --max-num-batched-tokens 2048 \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 3 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 48 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --enable-chunked-prefill=True \
        --disable-custom-all-reduce \
        --max-log-len 0 \
        --port 8862 > vllm.R1.log.3 2>&1 &
```

- error

```bash
 quantized or ignored modules [repeated 22x across cluster]
(RayWorkerWrapper pid=269885,

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]:  deepseek-coder-v2-lite-instruct;  Exception in worker VllmWorkerProcess while processing method initialize_cache: [Errno 2] No such file or directory: '/root/.triton/cache/de758c429c9ff1f18930bbd9c3004506/fused_moe_kernel.json.tmp.pid_1528_587007', Traceback (most recent call last):

**Link**: https://github.com/vllm-project/vllm/issues/6276
**State**: closed
**Created**: 2024-07-10T01:32:01+00:00
**Closed**: 2024-11-25T02:04:50+00:00
**Comments**: 10
**Labels**: bug, stale

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
CMake version: version 3.30.0
Libc version: glibc-2.31

Python version: 3.9.2 (default, Feb 28 2021, 17:03:44)  [GCC 10.2.1 20210110] (64-bit runtime)
Python platform: Linux-5.4.143.bsk.8-amd64-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L40
GPU 1: NVIDIA L40
GPU 2: NVIDIA L40
GPU 3: NVIDIA L40

Nvidia driver version: Could not collect
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.0
/usr/li

[... truncated for brevity ...]

---

## Issue #N/A: AWQ + Marlin Error

**Link**: https://github.com/vllm-project/vllm/issues/3392
**State**: closed
**Created**: 2024-03-14T04:52:22+00:00
**Closed**: 2024-07-25T19:53:41+00:00
**Comments**: 16

### Description

I convert model follow AutoAWQ library as follow script.

1. Quantize with Marlin
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
quant_path = 'mistral-instruct-v0.2-awq-marlin'
quant_config = { "zero_point": False, "q_group_size": 128, "w_bit": 4, "version": "Marlin" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
```

2. Generate

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer


quant_path = "./mistral-instruct-v0.2-awq-marlin"

# Load model
m

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Why does torch.cuda.memory_allocated() remain unchanged after calling sleep()?

**Link**: https://github.com/vllm-project/vllm/issues/17117
**State**: open
**Created**: 2025-04-24T15:38:59+00:00
**Comments**: 6
**Labels**: bug

### Description

### Your current environment

Hi! Although the free bytes increase (as shown by torch.cuda.mem_get_info()[0]), why does torch.cuda.memory_allocated() remain unchanged after calling sleep()?

<details>
<summary>See myexperience</summary>

```text
(Pdb) print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
Allocated: 28294.25 MB
(Pdb) print(f"Free bytes: {torch.cuda.mem_get_info()[0]/ 1e6:.2f} MB")
Free bytes: 79872.92 MB
(Pdb) print(f"Reserved : {torch.cuda.memory_reserved() / 1e6:.2f} MB")
Reserved : 29047.65 MB
(Pdb) self.llm.wake_up()
INFO 04-24 14:05:33 executor_base.py:219] It took 0.298841 seconds to wake up.
(Pdb) print(f"Reserved : {torch.cuda.memory_reserved() / 1e6:.2f} MB")
Reserved : 29047.65 MB
(Pdb) print(f"Free bytes: {torch.cuda.mem_get_info()[0]/ 1e6:.2f} MB")
Free bytes: 54379.94 MB
(Pdb) print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
Allocated: 28294.25 MB
(Pdb) self.llm.sleep(level=2)
INFO 04-24 14:05:59 worker.py:133] Sleep mode freed 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Gloo Connection reset by peer

**Link**: https://github.com/vllm-project/vllm/issues/6308
**State**: closed
**Created**: 2024-07-10T14:28:48+00:00
**Closed**: 2024-07-12T14:10:41+00:00
**Comments**: 15
**Labels**: bug

### Description

### Your current environment
```text
Collecting environment information...
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.1 LTS (x86_64)
GCC version: (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0
Clang version: Could not collect
CMake version: version 3.30.0
Libc version: glibc-2.35

Python version: 3.10.6 (main, Nov 14 2022, 16:10:14) [GCC 11.3.0] (64-bit runtime)
Python platform: Linux-5.15.0-58-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L4
GPU 1: NVIDIA L4
GPU 2: NVIDIA L4
GPU 3: NVIDIA L4
GPU 4: NVIDIA L4
GPU 5: NVIDIA L4
GPU 6: NVIDIA L4
GPU 7: NVIDIA L4

Nvidia driver version: 535.86.10
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True


Versions of relevant libraries:
[pip3] n

[... truncated for brevity ...]

---

## Issue #N/A: Long context will cause the vLLM stop

**Link**: https://github.com/vllm-project/vllm/issues/286
**State**: closed
**Created**: 2023-06-28T07:07:46+00:00
**Closed**: 2023-11-05T01:30:47+00:00
**Comments**: 12
**Labels**: bug

### Description

If I exceed the token limit of 4096, the vLLM abruptly stops. It would be helpful if you could incorporate some logging functionality into the stopping code. This way, users can easily modify the code to resume the vLLM from where it left off.

---

## Issue #N/A: [Feature]: Make SequenceClassification(Qwen3ForSequenceClassification) models support auto prefix cache

**Link**: https://github.com/vllm-project/vllm/issues/20894
**State**: open
**Created**: 2025-07-14T02:01:16+00:00
**Comments**: 9
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

As more and more LLMs are used for classification tasks, can we consider supporting this feature?

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: vllm always tries to download model from huggingface/modelscope even if I specify --download-dir  with already downloaded models

**Link**: https://github.com/vllm-project/vllm/issues/1755
**State**: closed
**Created**: 2023-11-22T14:06:46+00:00
**Closed**: 2024-04-03T15:27:58+00:00
**Comments**: 7

### Description

The situation: I've downloaded the huge models on my server. And hope vllm could load the model.

the structure of the model dir: 
``` $ ls /data/vllm.model/01ai/Yi-34B-200K/ ```

```
LICENSE              generation_config.json            pytorch_model-00004-of-00007.bin  tokenizer.json
README.md            md5                               pytorch_model-00005-of-00007.bin  tokenizer.model
Yi.svg               modeling_yi.py                    pytorch_model-00006-of-00007.bin  tokenizer_config.json
config.json          pytorch_model-00001-of-00007.bin  pytorch_model-00007-of-00007.bin
configuration.json   pytorch_model-00002-of-00007.bin  pytorch_model.bin.index.json
configuration_yi.py  pytorch_model-00003-of-00007.bin  tokenization_yi.py
```

Try to load the mode as:
```
VLLM_USE_MODELSCOPE=True python -m vllm.entrypoints.openai.api_server --model 01ai/Yi-34B-200K  --download-dir /data/vllm.model/ --host 0.0.0.0   --trust-remote-code
```

But it always tries to dow

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Error When Launching Llama-4-Scout-17B-16E-Instruct Without `--kv-cache-dtype fp8`

**Link**: https://github.com/vllm-project/vllm/issues/16150
**State**: closed
**Created**: 2025-04-07T03:33:28+00:00
**Closed**: 2025-04-15T06:11:13+00:00
**Comments**: 6
**Labels**: bug, torch.compile

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 04-07 11:13:31 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
/usr/local/lib/python3.10/dist-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.28.1
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-94-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.3.107
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80G

[... truncated for brevity ...]

---

## Issue #N/A: Local models cannot be used when the network is not accessible.

**Link**: https://github.com/vllm-project/vllm/issues/1824
**State**: closed
**Created**: 2023-11-29T03:40:15+00:00
**Closed**: 2024-04-03T15:26:51+00:00
**Comments**: 8

### Description

2023-11-29 11:34:19 Traceback (most recent call last):
2023-11-29 11:34:19   File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 467, in _make_request
2023-11-29 11:34:19     self._validate_conn(conn)
2023-11-29 11:34:19   File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 1096, in _validate_conn
2023-11-29 11:34:19     conn.connect()
2023-11-29 11:34:19   File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 642, in connect
2023-11-29 11:34:19     sock_and_verified = _ssl_wrap_socket_and_match_hostname(
2023-11-29 11:34:19   File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 782, in _ssl_wrap_socket_and_match_hostname
2023-11-29 11:34:19     ssl_sock = ssl_wrap_socket(
2023-11-29 11:34:19   File "/usr/local/lib/python3.10/dist-packages/urllib3/util/ssl_.py", line 470, in ssl_wrap_socket
2023-11-29 11:34:19     ssl_sock = _ssl_wrap_socket_impl(sock, context, tls_in_tls, server_

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Qwen 3 MoE Lora adapter support.

**Link**: https://github.com/vllm-project/vllm/issues/18120
**State**: closed
**Created**: 2025-05-14T06:50:30+00:00
**Closed**: 2025-07-17T18:32:53+00:00
**Comments**: 16
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

#### Feature Proposal:
Support for **Qwen 3 MoE LoRA (Low-Rank Adaptation)** adapter in vLLM to enable efficient fine-tuning and inference.

#### Motivation:
The Qwen 3 MoE model offer very good capabilities and performance. However, current vLLM do not support integration with LoRA adapters for fine-tuning and serving multiple finetuned models.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: Gemma3 vllm serve AttributeError: 'Gemma3Config' object has no attribute 'vocab_size'

**Link**: https://github.com/vllm-project/vllm/issues/14687
**State**: closed
**Created**: 2025-03-12T16:53:52+00:00
**Closed**: 2025-03-12T17:45:28+00:00
**Comments**: 6
**Labels**: usage

### Description

### Your current environment

```text
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.1 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.39

Python version: 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-6.8.0-52-generic-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 12.0.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090

Nvidia driver version: 550.120
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        48 bits physical, 48 bits virtual
Byte Order:            

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: LoRA support for InternVLChatModel

**Link**: https://github.com/vllm-project/vllm/issues/9495
**State**: closed
**Created**: 2024-10-18T08:03:53+00:00
**Closed**: 2025-03-05T02:02:37+00:00
**Comments**: 17
**Labels**: feature request, stale

### Description

### Your current environment

vllm version = 0.6.1

### Model Input Dumps

_No response_

### 🐛 Describe the bug

<details>
<summary>The output of `command:`</summary>


vllm version = 0.6.1. InternVLChat is in list of supported models.

```
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server --model OpenGVLab/InternVL2-8B --vllm_enable_lora=True --vllm_max_lora_rank=32 --lora-modules line_items=checkpoint-786/ --api-key=abcd  --host=0.0.0.0 --port=8817 --gpu_memory_utilization 0.95 --max_model_len=8192 --trust_remote_code --limit-mm-per-prompt 'image=16' 
```

```
rank0]:   File "/root/anaconda3/envs/msswift_latest/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py", line 636, in __init__
[rank0]:     self.engine = self._init_engine(*args, **kwargs)
[rank0]:   File "/root/anaconda3/envs/msswift_latest/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py", line 840, in _init_engine
[rank0]:     return engine_class(*args, **kwar

[... truncated for brevity ...]

---

## Issue #N/A: Unable to run baichuan13b on 2 GPUs

**Link**: https://github.com/vllm-project/vllm/issues/566
**State**: closed
**Created**: 2023-07-25T04:50:32+00:00
**Closed**: 2023-08-07T21:38:33+00:00
**Comments**: 6

### Description

Hi, I have 2 4090, each one of them can not fully load a 13b model, but vllm unable to automatically locate model into 2 GPUs, what else need I specific? (I have set a 0.8 GPU frac due to one GPU have a tiny process running consumes about  1GB mem)

---

## Issue #N/A: [Bug]: Missing metrics  in V1

**Link**: https://github.com/vllm-project/vllm/issues/16348
**State**: open
**Created**: 2025-04-09T14:38:15+00:00
**Comments**: 13
**Labels**: bug, v1

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 3.18.4
Libc version: glibc-2.31

Python version: 3.10.16 | packaged by conda-forge | (main, Dec  5 2024, 14:16:10) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.10.0-34-cloud-amd64-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L4
GPU 1: NVIDIA L4

Nvidia driver version: 550.90.07
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Byte Order:   

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Embedding model with Lora doesn't work.

**Link**: https://github.com/vllm-project/vllm/issues/12808
**State**: closed
**Created**: 2025-02-06T05:55:48+00:00
**Closed**: 2025-03-18T12:07:01+00:00
**Comments**: 7
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 02-06 16:43:14 __init__.py:183] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: SUSE Linux Enterprise Server 15 SP4 (x86_64)
GCC version: (GCC) 12.3.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.31

Python version: 3.10.16 | packaged by conda-forge | (main, Dec  5 2024, 14:16:10) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.14.21-150400.24.28-default-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.2.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H100
GPU 1: NVIDIA H100
GPU 2: NVIDIA H100
GPU 3: NVIDIA H100

Nvidia driver version: 550.54.15
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: 

[... truncated for brevity ...]

---

## Issue #N/A: pip install error - CUDA version mismatch

**Link**: https://github.com/vllm-project/vllm/issues/763
**State**: closed
**Created**: 2023-08-15T11:46:42+00:00
**Closed**: 2023-08-25T08:58:30+00:00
**Comments**: 9
**Labels**: installation

### Description

I have the torch version 2.0.1+cu117

```
import torch
print(torch.__version__)
print(torch.version.cuda)
2.0.1+cu117
11.7
```
However pip install vllm fails with
```

File "/tmp/pip-build-env-jw9yras7/overlay/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 387, in _check_cuda_version
          raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
      RuntimeError:
      The detected CUDA version (12.2) mismatches the version that was used to compile
      PyTorch (11.7). Please make sure to use the same CUDA versions.
      
      [end of output]
```
      
nvidia-smi
```
Tue Aug 15 11:43:53 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Unco

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: How to use intel-gpu in openvino

**Link**: https://github.com/vllm-project/vllm/issues/7418
**State**: closed
**Created**: 2024-08-12T13:43:15+00:00
**Closed**: 2025-01-11T01:59:57+00:00
**Comments**: 7
**Labels**: misc, stale

### Description

### Anything you want to discuss about vllm.

Hi, I successfully create the openvino env. I am wondering how to use the intel-gpu? 

---

## Issue #N/A: GPTQ does not support bfloat16

**Link**: https://github.com/vllm-project/vllm/issues/2149
**State**: closed
**Created**: 2023-12-17T06:06:30+00:00
**Closed**: 2024-11-30T02:03:24+00:00
**Comments**: 6
**Labels**: help wanted, feature request, quantization, stale

### Description

Currently, our GPTQ kernels only support the float16 precision.

---

## Issue #N/A: [Usage]: V0 Does Qwen2-VL Support torch.compile in vllm?

**Link**: https://github.com/vllm-project/vllm/issues/12693
**State**: closed
**Created**: 2025-02-03T14:31:02+00:00
**Closed**: 2025-06-20T02:13:44+00:00
**Comments**: 12
**Labels**: usage, stale

### Description

### Your current environment

Hi there,

First of all, thank you for all the hard work on vllm—it’s an excellent project!

I am currently exploring the use of torch.compile within vllm to optimize inference performance. I have seen that many decoder-only models (such as GPT-series and LLaMA) work well with torch.compile. However, I am particularly interested in using the Qwen2-VL model and could not find any documentation or discussion regarding torch.compile support for it.

Could you please clarify the following:

Is Qwen2-VL currently supported with torch.compile in the latest version of vllm?
If not, are there any plans to add support for Qwen2-VL with torch.compile in the near future?
Are there any known workarounds or tips for using torch.compile with multi-modal models like Qwen2-VL?
Any guidance or insights would be greatly appreciated!

Thank you for your time and assistance.

### How would you like to use vllm

I want to run inference of a [specific model](put link here). I d

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: ImportError: _flash_supports_window_size missing for baichuan-inc/Baichuan-M1-14B-Instruct (with trust_remote_code=True) in vLLM v0.8.2

**Link**: https://github.com/vllm-project/vllm/issues/15844
**State**: open
**Created**: 2025-04-01T02:30:43+00:00
**Comments**: 8
**Labels**: bug, stale

### Description

### Your current environment

환경 (Environment):

vLLM 버전 (vLLM Version): 0.8.2 (vllm/vllm-openai:v0.8.2 도커 이미지 사용)
Transformers 버전 (Transformers Version): vllm/vllm-openai:v0.8.2 이미지에 포함된 버전 (정확한 버전 확인 필요 시 명시, 현재는 오류로 미루어 호환되지 않는 버전으로 추정)
Python 버전 (Python Version): (로그에서 확인된 버전, 예: 3.12)
CUDA 버전 (CUDA Version): (vllm/vllm-openai:v0.8.2 이미지 기반 버전, 예: 11.8 또는 12.x)
사용 모델 (Model Used): baichuan-inc/Baichuan-M1-14B-Instruct
실행 환경 (Runtime Environment): Kubernetes Deployment (NVIDIA GPU 사용)
재현 단계 (Steps to Reproduce):

vllm/vllm-openai:v0.8.2 도커 이미지를 사용합니다.
다음과 유사한 명령어로 vLLM 서버를 시작합니다 (쿠버네티스 Deployment args 기준):
Bash

vllm serve baichuan-inc/Baichuan-M1-14B-Instruct \
    --trust-remote-code \
    --enable-chunked-prefill \
    --max_num_batched_tokens 1024


### 🐛 Describe the bug

관찰된 동작 (Observed Behavior):

vLLM 서버가 모델 로딩 단계에서 실패하며 다음과 같은 ImportError 로그를 남기고 컨테이너가 비정상 종료됩니다. 쿠버네티스 환경에서는 Pod가 READY: 0/1 상태가 되고 반복적으로 재시작합니다 (CrashLoopBackOff).


ERROR 03-31 18:56:05 [core.py:343] Engine

[... truncated for brevity ...]

---

## Issue #N/A: Phi-2 Broken

**Link**: https://github.com/vllm-project/vllm/issues/2422
**State**: closed
**Created**: 2024-01-11T21:35:22+00:00
**Closed**: 2024-02-12T20:00:09+00:00
**Comments**: 10

### Description

I've tried pip's version of `vllm`, and github `master`.
I've tried pip's version of `transformers` and github `master`.
I've tried an older `git clone` of HF's `microsoft/phi-2`, and then `git cloned` it anew.

When I try running it in python code, or in server mode, I get this error:

```sh
File "/home/mobius/_/lib/vllm/vllm/model_executor/models/phi_1_5.py", line 219, in __init__
    self.h = nn.ModuleList([
  File "/home/mobius/_/lib/vllm/vllm/model_executor/models/phi_1_5.py", line 220, in <listcomp>
    PhiLayer(config, linear_method)
  File "/home/mobius/_/lib/vllm/vllm/model_executor/models/phi_1_5.py", line 186, in __init__
    eps=config.layer_norm_epsilon)
  File "/home/mobius/_/lib/transformers/src/transformers/configuration_utils.py", line 265, in __getattribute__
    return super().__getattribute__(key)
AttributeError: 'PhiConfig' object has no attribute 'layer_norm_epsilon'. Did you mean: 'layer_norm_eps'?
```

If I go into the `vllm` code and fix that,

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM (running HuggingFaceTB/SmolVLM-Instruct) crashes with a 500 when making concurrent requests through the OpenAI compatible HTTP server

**Link**: https://github.com/vllm-project/vllm/issues/10931
**State**: closed
**Created**: 2024-12-05T16:44:15+00:00
**Closed**: 2024-12-06T12:18:36+00:00
**Comments**: 7
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 23.10 (x86_64)
GCC version: (Ubuntu 13.2.0-4ubuntu3) 13.2.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.38

Python version: 3.10.15 (main, Oct  3 2024, 07:27:34) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.5.0-44-generic-x86_64-with-glibc2.38
Is CUDA available: True
CUDA runtime version: 12.0.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090
GPU 2: NVIDIA GeForce RTX 4090
GPU 3: NVIDIA GeForce RTX 4090

Nvidia driver version: 535.171.04
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_6

[... truncated for brevity ...]

---

