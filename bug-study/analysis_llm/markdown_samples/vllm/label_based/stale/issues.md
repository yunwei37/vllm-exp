# stale - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 5
- Closed Issues: 25

### Label Distribution

- stale: 30 issues
- bug: 11 issues
- usage: 10 issues
- installation: 3 issues
- feature request: 2 issues
- performance: 2 issues
- ray: 1 issues
- x86-cpu: 1 issues
- quantization: 1 issues

---

## Issue #N/A: [Usage]: When running models on multiple GPUs, workload does not get split

**Link**: https://github.com/vllm-project/vllm/issues/12354
**State**: closed
**Created**: 2025-01-23T13:12:21+00:00
**Closed**: 2025-05-24T02:07:44+00:00
**Comments**: 2
**Labels**: usage, stale

### Description

### Your current environment

PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.11.6 (main, Nov  2 2023, 09:27:30) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-119-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA GeForce RTX 3090
GPU 1: NVIDIA GeForce RTX 3090
GPU 2: NVIDIA GeForce RTX 3090
GPU 3: NVIDIA GeForce RTX 3090

Nvidia driver version: 560.35.03
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

## Issue #N/A: How to use Splitwise(from microsoft) in vllm?

**Link**: https://github.com/vllm-project/vllm/issues/2370
**State**: closed
**Created**: 2024-01-08T03:49:36+00:00
**Closed**: 2024-11-30T02:03:07+00:00
**Comments**: 12
**Labels**: stale

### Description

Microsoft have claimed that ”Splitwise“ is supported in vLLM, see
https://www.microsoft.com/en-us/research/blog/splitwise-improves-gpu-usage-by-splitting-llm-inference-phases/
![image](https://github.com/vllm-project/vllm/assets/58217233/7835c241-f22c-4ffc-a510-1238f4a5d770)

So how to use it in vLLM? I could not find keyword about ”Splitwise“.

---

## Issue #N/A: [Usage]: Cannot use xformers with old GPU

**Link**: https://github.com/vllm-project/vllm/issues/10662
**State**: closed
**Created**: 2024-11-26T07:14:55+00:00
**Closed**: 2025-03-27T02:04:33+00:00
**Comments**: 9
**Labels**: usage, stale

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

OS: CentOS Linux 7 (Core) (x86_64)
GCC version: (GCC) 4.8.5 20150623 (Red Hat 4.8.5-44)
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.17

Python version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:27:36) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.10.0-1.0.0.32-x86_64-with-glibc2.17
Is CUDA available: True
CUDA runtime version: 12.4.131
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

Nvidia dr

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vllm using ray in eks hangs when using --pipeline_parallel_size > 1

**Link**: https://github.com/vllm-project/vllm/issues/11139
**State**: closed
**Created**: 2024-12-12T14:07:32+00:00
**Closed**: 2025-07-09T02:16:16+00:00
**Comments**: 10
**Labels**: bug, ray, stale

### Description

### Your current environment

running on a pod in g6.12xlarge (allocated by lws).
Pod is initializing ray before running vllm (using the proposed lws image https://github.com/kubernetes-sigs/lws/blob/main/docs/examples/vllm/build/Dockerfile.GPU)

### Model Input Dumps

_No response_

### 🐛 Describe the bug

Vllm is stuck on this meesage:
INFO 12-12 05:28:31 pynccl.py:69] vLLM is using nccl==2.21.5

full log:
[2024-12-12 05:27:53,632 W 8 8] global_state_accessor.cc:463: Retrying to get node with node ID 9c36d691ad808fe6b12015dc3c0c4ba0432917a72547d5450434659c
2024-12-12 05:27:52,822 INFO usage_lib.py:467 -- Usage stats collection is enabled by default without user confirmation because this terminal is detected t
o be non-interactive. To disable this, add `--disable-usage-stats` to the command that starts the cluster, or run the following command: `ray disable-usage
-stats` before starting the cluster. See https://docs.ray.io/en/master/cluster/usage-stats.html for more details.


[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Tools parsing issues with mistral3.1

**Link**: https://github.com/vllm-project/vllm/issues/15549
**State**: open
**Created**: 2025-03-26T13:38:21+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

vllm 0.8.1


### 🐛 Describe the bug

seems there is an issue with mistral for tools parsing? the output is not function calling as expected.

- command:
`serve mistralai/Mistral-Small-3.1-24B-Base-2503 --max-model-len 4096 --gpu-memory-utilization 0.9 --tensor-parallel-size 4 --served-model-name mistral --tokenizer-mode mistral --config-format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice `

example:
- request:
```
 {
    "model":"mistral",
    "messages": [
        {
            "content": "What's the weather like in San Francisco?",
            "role": "user"
        }
    ],
    "max_completion_tokens": 128,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
        

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Mismatch in the number of image tokens and placeholders during batch inference

**Link**: https://github.com/vllm-project/vllm/issues/7669
**State**: closed
**Created**: 2024-08-20T01:21:26+00:00
**Closed**: 2024-12-28T01:59:21+00:00
**Comments**: 14
**Labels**: bug, stale

### Description

### Your current environment

```
Ray v2.23
Python 3.10
vllm 0.5.4
cuda 12.1
```

### 🐛 Describe the bug

We are attempting to utilize Ray v2.23 for batch inferencing, specifically on multi-modal data, by leveraging llava-next. 

```
dataset = ray.data.read_parquet(gcsInputPath, columns=columns)
class LLMPredictor:

    def __init__(self):
        # Create an LLM.
        self.llm = LLM(model="/mnt/models",
                       tensor_parallel_size=1)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:

        try:
            start_time = time.time()

            prompts = [{"prompt": prompt, "multi_modal_data": {
                "image": Image.open(io.BytesIO(base64.b64decode(batch[imageColumnName][i])))}} for i in
                       range(len(batch[imageColumnName]))]

            predictions = self.llm.generate(
                prompts, sampling_params=sampling_params)
            batch["generated_output"] = [preds.outpu

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: VLLM Inference - 2x slower with LoRA rank=256 vs none.

**Link**: https://github.com/vllm-project/vllm/issues/14435
**State**: open
**Created**: 2025-03-07T11:58:26+00:00
**Comments**: 14
**Labels**: usage, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

I've noticed that using LoRA with rank=256 significantly slows down inference by 4x, as shown below. However, reducing the rank to 8 or 16 brings performance closer to that of no LoRA. I'm currently using two fully-utilized GPUs, without the enforce_eager flag, and have set the maximum LoRA rank accordingly. Interestingly, adjusting the maximum model length had no impact on performance. What steps can I take to optimize performance?


**No Lora**

**Processed prompts**:   0%|▏                                                            | 5/2430 [01:28<6:58:39, 10.36s/it, est. speed input: 3.71 toks/s, output: 2.34 toks/s]Processed prompts:  10%|█████▊                                                     | 240/2430 [05:09<44:09,  1.21s/it, est. speed input: 87.79 toks/s, output: 90.18 toks/s]WARNING 03-06 17:12:30 scheduler.py:1754] Sequence group 352 is preempted by Preem

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: control over llm_engine placement when multiple gpus are available.

**Link**: https://github.com/vllm-project/vllm/issues/6312
**State**: closed
**Created**: 2024-07-10T16:01:34+00:00
**Closed**: 2024-11-25T02:04:39+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

I need a way to specify which gpu exactly should vllm use when multiple gpus are available. Currently, it automatically occupies all available gpus (https://docs.vllm.ai/en/latest/serving/distributed_serving.html).

For example, something like this: `vllm.LLM(model_path, device="cuda:N")`

#691 is exactly the same question but they end up agreeing that they can use Ray. I'm asking for a simpler solution that would not require spending time on extra engineering.

### Alternatives

My use-case doesn't allow me to use CUDA_VISIBLE_DEVICES to specify which gpu to use. That's because i train a model on multiple gpus in a DDP-like fashion where each vllm instance generates data for a model on its device, then gradients are synchronized and so on. So I cannot set CUDA_VISIBLE_DEVICES to some specific device as that would turn multiple-gpu training in a single-gpu training.

Also, I cannot just avoid this problem by running a vllm-server on a sepa

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: can't get the cu118 version of vllm 0.6.3 by https://github.com/vllm-project/vllm/releases/download/v0.6.3/vllm-0.6.3+cu118-cp310-cp310-manylinux1_x86_64.whl

**Link**: https://github.com/vllm-project/vllm/issues/10540
**State**: closed
**Created**: 2024-11-21T14:28:49+00:00
**Closed**: 2025-03-22T02:02:48+00:00
**Comments**: 2
**Labels**: installation, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How you are installing vllm

```sh
pip install -vvv vllm
```


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: Can I get the loss of model directly?

**Link**: https://github.com/vllm-project/vllm/issues/9750
**State**: open
**Created**: 2024-10-28T08:05:33+00:00
**Comments**: 6
**Labels**: usage, stale

### Description

Hi, great work!
I am currently optimizing LLM based on `vLLM` and need to test whether my optimizations affect the model's perplexity. Therefore, I want to obtain the model's cross-entropy loss. I have reviewed the issue: [Can I directly obtain the logits here?](https://github.com/vllm-project/vllm/issues/185) and understand that one way to get log probabilities is by setting the `logprobs` parameter in `SampleParams`. 

However, this method is not very convenient. We can only obtain the top-n most likely log probabilities for each token, and the probability of the correct token might not be among these top-n log probabilities. Setting `n` and searching for the probability of the correct token is quite cumbersome, and the cross-entropy has to be calculated manually as well. 

Therefore, I want to know if `vLLM` has a way to directly obtain cross-entropy, similar to `transformers`. 
Thank you sincerely for your help. :-)

---

## Issue #N/A: [Performance]: Issues with prefix cache usage 

**Link**: https://github.com/vllm-project/vllm/issues/8005
**State**: closed
**Created**: 2024-08-29T16:49:45+00:00
**Closed**: 2024-12-29T02:05:16+00:00
**Comments**: 2
**Labels**: performance, stale

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

I have been using vLLM with prefix caching to optimise inference in cases where majority of operations are pre-fills with large shared prefix. Specially, most of the prompts are 130 tokens in size with 90% of it is a shared system prompt. 
The decode is only phase is only one token.
There benchmark is a 100000 prompts (`formatted_prompts` below) executed via generate:
```python
from outlines import models, generate
llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct", enable_prefix_caching=True)
sampling_params = SamplingParams(temperature=0.5, top_p=0.2, max_tokens=1)
model = models.VLLM(llm)
generator = generate.choice(model, ["yes", "no"])
predictions = generator(formatted_prompts, sampling_params=sampling_params)
``` 

During experiments I have observed that if I use the **same prompt** repeatedly (`formatted_prompts` is identical prompt repeated 100000 times) I observe **no throu

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: 请求报错

**Link**: https://github.com/vllm-project/vllm/issues/8755
**State**: closed
**Created**: 2024-09-24T01:55:35+00:00
**Closed**: 2025-01-24T01:58:50+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

vllm  0.5.1
2*A100
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 7861 --max-model-len 9000 --served-model-name chat-v2.0 --model /workspace/sdata/qwen2-72B-instruct --enforce-eager  --tensor-parallel-size 2 --gpu-memory-utilization 0.95

### Model Input Dumps

![image](https://github.com/user-attachments/assets/c0fee9bc-8ff9-414c-a89a-e004ebd51f6e)

### 🐛 Describe the bug

请求报错

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: Bug in quantization/awq /gemm_kernels.cu gemm_forward_4bit_cuda_m16nXk32 More result have been write

**Link**: https://github.com/vllm-project/vllm/issues/7400
**State**: closed
**Created**: 2024-08-11T14:08:06+00:00
**Closed**: 2024-12-12T02:06:52+00:00
**Comments**: 6
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

<img width="774" alt="截屏2024-08-11 下午10 05 10" src="https://github.com/user-attachments/assets/968487f3-a1e4-45ef-8fc4-d8f00a07c2bd">

When N=64, we don't have 4*8=32 c_warp result; In this case, we only have 2(N/32) * 8=16 c_warp results.

---

## Issue #N/A: [Installation]: May I ask if there is a good solution for deploying grmma-2-27b on v100? The deployment has been consistently unsuccessful

**Link**: https://github.com/vllm-project/vllm/issues/11462
**State**: closed
**Created**: 2024-12-24T09:41:08+00:00
**Closed**: 2025-04-25T02:08:29+00:00
**Comments**: 5
**Labels**: installation, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How you are installing vllm

```sh
pip install -vvv vllm
```


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Performance]: Poor performance of vllm on AWQ

**Link**: https://github.com/vllm-project/vllm/issues/3581
**State**: closed
**Created**: 2024-03-23T11:15:31+00:00
**Closed**: 2024-11-29T02:07:15+00:00
**Comments**: 2
**Labels**: performance, stale

### Description

### Proposal to improve performance

https://github.com/InternLM/lmdeploy/tree/main
This project is twice faster than vllm with AWQ int4

### Report of performance regression

_No response_

### Misc discussion on performance

_No response_

### Your current environment (if you think it is necessary)

_No response_

---

## Issue #N/A: [Bug]: vllm server bad

**Link**: https://github.com/vllm-project/vllm/issues/13340
**State**: closed
**Created**: 2025-02-16T01:45:55+00:00
**Closed**: 2025-06-18T02:13:26+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

vllm 0.7.2
torch 2.4
cuda 12.1


### 🐛 Describe the bug

OpenAI-Compatible Server in chat window call by url base_url="http://localhost:8000/v1"  when call api，why 200 OK only the first time and then always 400 Bad Request：

log 阿斯follows：
INFO:     127.0.0.1:59042 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 02-15 22:09:59 engine.py:275] Added request chatcmpl-803293759b1e415caefd7845b3fa8352.
INFO 02-15 22:10:03 metrics.py:455] Avg prompt throughput: 33.4 tokens/s, Avg generation throughput: 37.8 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.1%, CPU KV cache usage: 0.0%.
INFO 02-15 22:10:08 metrics.py:455] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 43.2 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.1%, CPU KV cache usage: 0.0%.
INFO:     127.0.0.1:59042 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request


### Before submitting a new issue...

- [x] Ma

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Execution speed of non-Lora requests

**Link**: https://github.com/vllm-project/vllm/issues/8368
**State**: closed
**Created**: 2024-09-11T11:48:29+00:00
**Closed**: 2025-01-24T01:59:03+00:00
**Comments**: 8
**Labels**: usage, stale

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.31

Python version: 3.10.14 (main, Apr  6 2024, 18:45:05) [GCC 9.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-116-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A100-SXM4-80GB
Nvidia driver version: 535.183.01
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Byte Order:                           Little End

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: vllm: error: unrecognized arguments: --lora-path

**Link**: https://github.com/vllm-project/vllm/issues/13669
**State**: closed
**Created**: 2025-02-21T12:45:10+00:00
**Closed**: 2025-06-23T02:14:59+00:00
**Comments**: 13
**Labels**: usage, stale

### Description

### Your current environment

```
INFO 02-21 12:37:49 __init__.py:207] Automatically detected platform cuda.
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

Python version: 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.4.0-167-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 11.8.89
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A40
GPU 1: NVIDIA A40

Nvidia driver version: 565.57.01
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]:  benchmark_throughput gets TypeError: XFormersMetadata.__init__() got an unexpected keyword argument 'is_prompt' wit CPU 

**Link**: https://github.com/vllm-project/vllm/issues/6225
**State**: closed
**Created**: 2024-07-08T21:58:11+00:00
**Closed**: 2025-03-14T02:02:55+00:00
**Comments**: 18
**Labels**: bug, x86-cpu, stale

### Description

### Your current environment

```text
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.0
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-91-generic-x86_64-with-glibc2.35

...

[pip3] numpy==1.26.4
[pip3] nvidia-nccl-cu12==2.20.5
[pip3] torch==2.3.0
[pip3] torchvision==0.18.0
[pip3] transformers==4.42.3
[pip3] triton==2.3.0
[conda] numpy                     1.26.4                   pypi_0    pypi
[conda] nvidia-nccl-cu12          2.20.5                   pypi_0    pypi
[conda] torch                     2.3.0                    pypi_0    pypi
[conda] torchvision               0.18.0                   pypi_0    pypi
[conda] transformers              4.42.3       

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Inconsistent Output Behavior with and without tools and tool_choice Parameters

**Link**: https://github.com/vllm-project/vllm/issues/7693
**State**: closed
**Created**: 2024-08-20T13:45:51+00:00
**Closed**: 2024-12-21T01:58:53+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment
v0.5.4

In the VLLM server setup, specifying tools and tool_choice parameters produces a direct output, while omitting them leads to a descriptive response about the intended function call. This inconsistency arises regardless of identical token inputs, highlighting a potential issue in handling these parameters.

### 🐛 Describe the bug

This is how my prompt looks after applying chat template:
```
<s> <tools> You have access to a range of tools designed to assist you with various tasks. These tools enable you to perform specific functions and provide more precise and effective responses. Here are the tools you can utilize: <ul>\n <li> {'name': 'get_current_weather', 'description': 'Get the current weather', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'}}, 'required': ['location']}} </li>\n</ul> </tools>[INST] What is the weather for Istanbul? [/INST]
```

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: The service operation process results in occasional exception errors RuntimeError: CUDA error: an illegal memory access was encountered

**Link**: https://github.com/vllm-project/vllm/issues/11366
**State**: closed
**Created**: 2024-12-20T09:01:14+00:00
**Closed**: 2025-04-20T02:10:42+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Alibaba Group Enterprise Linux Server 7.2 (Paladin) (x86_64)
GCC version: (GCC) 9.2.1 20200522 (Alibaba 9.2.1-3 2.17)
Clang version: Could not collect
CMake version: version 3.30.5
Libc version: glibc-2.30

Python version: 3.8.18 (default, Sep 11 2023, 13:40:15)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-4.9.151-015.ali3000.alios7.x86_64-x86_64-with-glibc2.17
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA L20
Nvidia driver version: 535.161.08
cuDNN version: Probably one of the following:
/usr/local/cuda/targets/x86_64-linux/lib/libcudnn.so.8.9.3
/usr/local/cuda/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.9.3
/usr/local/cuda/target

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: n_inner divisible to number of GPUs

**Link**: https://github.com/vllm-project/vllm/issues/3772
**State**: closed
**Created**: 2024-04-01T09:21:34+00:00
**Closed**: 2024-11-28T02:07:10+00:00
**Comments**: 5
**Labels**: bug, stale

### Description

### Your current environment

I was using the latest docker image(0.4.0) with 4-8L4 GPUs for the mentioned problem. I also tested this with installing from source as well with a custom docker image.

### 🐛 Describe the bug

Hello, first of all, thank you for the grand work!

I was trying to utilize the recently supported JAIS models. When I try [jais-30b-chat-v3](https://huggingface.co/core42/jais-30b-chat-v3) with 8xL4 GPUs, I was getting the error

```bash
... AssertionError: 19114 is not divisible by 8 [repeated 2x across cluster]
```

I wanted to test the [jais-13b-chat](https://huggingface.co/core42/jais-13b-chat) model for the same purpose to see if I can deploy it to 4xL4 GPUs and I got

```bash
... AssertionError: 13653 is not divisible by 4 [repeated 2x across cluster]
```

Commands that I was utilizing can be generalized along the lines of:

```bash

MODEL=core42/jais-30b-chat-v3
NUM_GPUS=8
docker run --runtime nvidia --gpus all \
    -v ~/.cache/hu

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: How to inference a model with medusa speculative sampling.

**Link**: https://github.com/vllm-project/vllm/issues/6768
**State**: closed
**Created**: 2024-07-25T04:02:27+00:00
**Closed**: 2025-01-19T02:01:57+00:00
**Comments**: 4
**Labels**: usage, stale

### Description

### Your current environment

Collecting environment information...
PyTorch version: 2.3.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.0
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-125.006-nvidia-x86_64-with-glibc2.35
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

Nvidia driver version: 550.90.07
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen ru

[... truncated for brevity ...]

---

## Issue #N/A: Quantization for V100

**Link**: https://github.com/vllm-project/vllm/issues/1345
**State**: closed
**Created**: 2023-10-13T16:44:17+00:00
**Closed**: 2024-12-01T02:16:03+00:00
**Comments**: 12
**Labels**: quantization, stale

### Description

Similar to #1252 , do we have any plans for supporting V100. For now I can see that the place need to be modified is ldmatrix instruction and m16n8k16, as an example we may need to load the matrix manually and perform the mma in a smaller size, for example, maybe we need something similar to these
```c++
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
          // Manually loading each fragment, ldmatrix only available on sm_75 and after
          __asm__ __volatile__(
              "ld.shared.b16 %0, [%4];\n"
              "ld.shared.b16 %1, [%4 + 2];\n"
              "ld.shared.b16 %2, [%4 + 4];\n"
              "ld.shared.b16 %3, [%4 + 6];\n"
              : "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[0]), 
                "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[1]), 
                "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[2]), 
                "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr)
          );
#else


[... truncated for brevity ...]

---

## Issue #N/A: Include Llama-405B in nightly benchmarks?

**Link**: https://github.com/vllm-project/vllm/issues/7761
**State**: closed
**Created**: 2024-08-21T23:27:51+00:00
**Closed**: 2024-12-22T02:04:29+00:00
**Comments**: 5
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

Include the Llama-405B model as part of the nightly performance benchmarks here: https://buildkite.com/vllm/performance-benchmark/builds/4068

Is the reason for not doing so primarily cost (16-24 H100s needed)? If so Akash.Network would consider providing the infra for it. 

Thanks!

### Alternatives

Running the benchmarks ourselves 

### Additional context

We’ve been trying to run Llama-405B-FP8 in production (with vLLM + Ray) and have been encountering stability issues with it. 

---

## Issue #N/A: [Usage]: Compilation and Execution Issues Across Different GPU Models After Modifying vLLM Source Code

**Link**: https://github.com/vllm-project/vllm/issues/11914
**State**: closed
**Created**: 2025-01-10T02:49:36+00:00
**Closed**: 2025-05-17T02:09:03+00:00
**Comments**: 3
**Labels**: usage, stale

### Description

### Your current environment

```text
No module named 'vllm._version'
  from vllm.version import __version__ as VLLM_VERSION
Collecting environment information...
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.5
Libc version: glibc-2.35

Python version: 3.10.12 (main, Sep 11 2024, 15:47:36) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-3.10.0-1062.el7.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A100-PCIE-40GB
Nvidia driver version: 535.161.07
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU op-mode(s):          

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: [swift with vllm and only using vllm serve]  leads to different result（10% diff）

**Link**: https://github.com/vllm-project/vllm/issues/16632
**State**: open
**Created**: 2025-04-15T03:59:34+00:00
**Comments**: 11
**Labels**: usage, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```
 **~ $ python collect_env.py**
INFO 04-15 11:43:18 __init__.py:190] Automatically detected platform cuda.
Collecting environment information...
/home/work/py39/lib/python3.9/site-packages/_distutils_hack/__init__.py:26: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")
PyTorch version: 2.5.1+cu118
Is debug build: False
CUDA used to build PyTorch: 11.8
ROCM used to build PyTorch: N/A

OS: CentOS Linux 7 (Core) (x86_64)
GCC version: (GCC) 12.1.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.17

Python version: 3.9.2 (default, Mar  3 2021, 20:02:32)  [GCC 7.3.0] (64-bit runtime)
Python platform: Linux-5.10.0-1.0.0.34-x86_64-with-glibc2.17
Is CUDA available: True
CUDA runtime version: 10.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA A10
GPU 1: NVIDIA A10

Nvidia driver version: 535.

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Mistral Nemo Instruct almost never returns JSON when using `guided_json`

**Link**: https://github.com/vllm-project/vllm/issues/7004
**State**: closed
**Created**: 2024-07-31T20:31:11+00:00
**Closed**: 2024-12-01T02:14:18+00:00
**Comments**: 4
**Labels**: bug, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
Collecting environment information...
PyTorch version: 2.3.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Amazon Linux 2 (x86_64)
GCC version: (GCC) 7.3.1 20180712 (Red Hat 7.3.1-17)
Clang version: Could not collect
CMake version: version 3.27.7
Libc version: glibc-2.26

Python version: 3.10.9 | packaged by conda-forge | (main, Feb  2 2023, 20:20:04) [GCC 11.3.0] (64-bit runtime)
Python platform: Linux-5.10.220-209.869.amzn2.x86_64-x86_64-with-glibc2.26
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA L4
Nvidia driver version: 550.90.07
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little End

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: relationship between embedding size and vocab_size

**Link**: https://github.com/vllm-project/vllm/issues/15131
**State**: open
**Created**: 2025-03-19T14:02:44+00:00
**Comments**: 13
**Labels**: usage, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

I’ve noticed that the embedding size is always smaller than the vocab_size. Additionally, sometimes the `prompt_token_ids` are larger than the embedding size. ​Is there a way to map the embedding vector to each of the prompt tokens so that I can retrieve the logit of a prompt token like this:
`embeds[i, labels[i]]`?

```python
outputs = llm.encode(prompts)
print(f'vocab_size: {llm.get_tokenizer().vocab_size}')
for i in range(len(outputs)):
    labels = outputs[i].prompt_token_ids[1:]
    embeds = outputs[i].outputs.data
    print(f'{i}-th prompt_token_ids: {labels}')
    print(f'{i}-th embeddings: {embeds.shape}')
```

```log
Processed prompts: 100%|██████████| 4/4 [00:00<00:00, 55.18it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
vocab_size: 50254
0-th prompt_token_ids: [4007, 273, 253, 1986, 2077, 310]
0-th embeddings: torch.Size([7, 2560])
1-th prompt_token

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: Installing vllm in GH200 machine (aarch64) causes problems with cusparse.h missing

**Link**: https://github.com/vllm-project/vllm/issues/11191
**State**: closed
**Created**: 2024-12-14T02:18:26+00:00
**Closed**: 2025-04-18T02:06:38+00:00
**Comments**: 4
**Labels**: installation, stale

### Description

### Your current environment

I cannot run collect_env.py since that would require vllm
<img width="414" alt="Screenshot 2024-12-13 at 9 18 01 PM" src="https://github.com/user-attachments/assets/94ae6cdf-7a0b-4727-9650-d9f4d9599e4e" />



### How you are installing vllm

I am following the instructions from here:
[https://docs.vllm.ai/en/stable/getting_started/installation.html#use-an-existing-pytorch-installation](https://docs.vllm.ai/en/stable/getting_started/installation.html?fbclid=IwZXh0bgNhZW0CMTAAAR0rKk7-u-dGjP9zdYYSFVpbj0REfhwjOhFgzrLC2DWeQDb5D1KbQFy-xLQ_aem_aEMTM-Po9v5WOAzcqzVmlg#use-an-existing-pytorch-installation)

Problem I am facing:

```
pip install . --verbose --no-build-isolation
Using pip 24.3.1 from /work/nvme/bcfp/ftajwar/anaconda3/envs/exploration/lib/python3.10/site-packages/pip (python 3.10)
Processing /work/nvme/bcfp/ftajwar/vllm
  Running command Preparing metadata (pyproject.toml)
  running dist_info
  creating /tmp/pip-modern-metadata-nus2ddwg/v

[... truncated for brevity ...]

---

