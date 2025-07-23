# normal_1to7days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- bug: 10 issues
- usage: 7 issues
- feature request: 5 issues
- installation: 2 issues
- ray: 1 issues

---

## Issue #N/A: [Installation]: Can't find OpenMP headers on macOS

**Link**: https://github.com/vllm-project/vllm/issues/14034
**State**: closed
**Created**: 2025-02-28T10:20:31+00:00
**Closed**: 2025-03-03T01:35:02+00:00
**Comments**: 10
**Labels**: installation

### Description

Seems that clang can't find the OpenMP headers.

### Your current environment

```text
(vllm) ➜  vllm git:(v0.7.2) python collect_env.py 
INFO 02-28 18:13:24 __init__.py:190] Automatically detected platform cpu.
Collecting environment information...
PyTorch version: 2.5.1
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: macOS 15.3.1 (arm64)
GCC version: Could not collect
Clang version: 16.0.0 (clang-1600.0.26.6)
CMake version: version 3.31.5
Libc version: N/A

Python version: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 12:55:12) [Clang 14.0.6 ] (64-bit runtime)
Python platform: macOS-15.3.1-arm64-arm-64bit
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Apple M1 Max

Versions of relevant libraries:
[pip3] numpy==1.26

[... truncated for brevity ...]

---

## Issue #N/A: Support `ChatCompletion` Endpoint in OpenAI demo server

**Link**: https://github.com/vllm-project/vllm/issues/311
**State**: closed
**Created**: 2023-06-29T14:25:32+00:00
**Closed**: 2023-07-03T17:47:45+00:00
**Comments**: 4
**Labels**: feature request

### Description

@infwinston Feel free to use FastChat's completion template to implement a chat completion endpoint in our demo server. You can use the completion API as a reference:

https://github.com/vllm-project/vllm/blob/9d27b09d12767de775a92d765e177a61f8477189/vllm/entrypoints/openai/api_server.py#L88-L101

---

## Issue #N/A: [Bug]: vllm-0.4.0 is much slower than vllm-0.3.3

**Link**: https://github.com/vllm-project/vllm/issues/3779
**State**: closed
**Created**: 2024-04-01T17:33:48+00:00
**Closed**: 2024-04-02T22:32:22+00:00
**Comments**: 13

### Description

### Your current environment

Traceback (most recent call last):
  File "/zhanghongbo/collect_env.py", line 719, in <module>
    main()
  File "/zhanghongbo/collect_env.py", line 698, in main
    output = get_pretty_env_info()
  File "/zhanghongbo/collect_env.py", line 693, in get_pretty_env_info
    return pretty_str(get_env_info())
  File "/zhanghongbo/collect_env.py", line 530, in get_env_info
    vllm_version = get_vllm_version()
  File "/zhanghongbo/collect_env.py", line 262, in get_vllm_version
    return vllm.__version__
AttributeError: module 'vllm' has no attribute '__version__'

I encountered the above issue, so I commented out a line of code.

PyTorch version: 2.1.2+cu118
Is debug build: False
CUDA used to build PyTorch: 11.8
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.3 LTS (x86_64)
GCC version: (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Clang version: Could not collect
CMake version: version 3.26.1
Libc version: glibc-2.31

Python version: 3.9.1

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Return token strings in addition to token ids for /tokenize

**Link**: https://github.com/vllm-project/vllm/issues/18928
**State**: closed
**Created**: 2025-05-29T21:21:52+00:00
**Closed**: 2025-05-31T18:00:13+00:00
**Comments**: 0
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Currently /tokenize return array of token ids
```
"tokens": [
    50258,
    50363,
    16216
  ],
```
Can it also return the token strings, e.g.
```
"token_strs": [
    "this",
    " is",
    " a"
  ],
```

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Fail to build vllm from source for H100

**Link**: https://github.com/vllm-project/vllm/issues/3309
**State**: closed
**Created**: 2024-03-11T05:39:40+00:00
**Closed**: 2024-03-12T20:35:55+00:00
**Comments**: 6

### Description

It worked if I change `NVIDIA_SUPPORTED_ARCHS` in `setup.py` to `NVIDIA_SUPPORTED_ARCHS = {"8.0", "8.6"}`

```
#10 812.2       [1/12] /usr/local/cuda/bin/nvcc  -I/tmp/pip-build-env-o232h5cp/overlay/local/lib/python3.10/dist-packages/torch/include -I/tmp/pip-build-env-o232h5cp/overlay/local/lib/python3.10/dist-packages/torch/include/torch/csrc/api/include -I/tmp/pip-build-env-o232h5cp/overlay/local/lib/python3.10/dist-packages/torch/include/TH -I/tmp/pip-build-env-o232h5cp/overlay/local/lib/python3.10/dist-packages/torch/include/THC -I/usr/local/cuda/include -I/usr/include/python3.10 -c -c /home/corvo/vllm/csrc/cuda_utils_kernels.cu -o /tmp/tmp6jq49t5c.build-temp/csrc/cuda_utils_kernels.o --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80 --threads 8 -DENABLE_FP8_E5M2 -D__CUDA_NO_HALF_OPE

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: InternVL2-Llama3-76B-AWQ RUN ERROR KeyError: 'layers.39.mlp.gate_up_proj.qweight'

**Link**: https://github.com/vllm-project/vllm/issues/11122
**State**: closed
**Created**: 2024-12-12T03:59:26+00:00
**Closed**: 2024-12-17T02:44:39+00:00
**Comments**: 5
**Labels**: bug

### Description

### Your current environment

Neuron SDK Version: N/A
vLLM Version: 0.6.4.post1
vLLM Build Flags:
CUDA Archs: Not Set; ROCm: Disabled; Neuron: Disabled
GPU Topology:
GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      PIX     PXB     PXB     SYS     SYS     SYS     SYS     0-95            N/A             N/A
GPU1    PIX      X      PXB     PXB     SYS     SYS     SYS     SYS     0-95            N/A             N/A
GPU2    PXB     PXB      X      PXB     SYS     SYS     SYS     SYS     0-95            N/A             N/A
GPU3    PXB     PXB     PXB      X      SYS     SYS     SYS     SYS     0-95            N/A             N/A
GPU4    SYS     SYS     SYS     SYS      X      PIX     PXB     PXB     0-95            N/A             N/A
GPU5    SYS     SYS     SYS     SYS     PIX      X      PXB     PXB     0-95            N/A             N/A
GPU6    SYS     SYS     SYS     SYS     PXB     PXB      X      PX

[... truncated for brevity ...]

---

## Issue #N/A: Issue when trying to run inference on this model EleutherAI/gpt-j-6b

**Link**: https://github.com/vllm-project/vllm/issues/1563
**State**: closed
**Created**: 2023-11-04T18:29:30+00:00
**Closed**: 2023-11-08T07:12:38+00:00
**Comments**: 2

### Description

EleutherAI/gpt-j-6b is mentioned as supported in the docs. Trying to run inference on Google Colab with free tier GPU. Getting this error. AssertionError: tensor model parallel group is already initialized.

---

## Issue #N/A: [Usage]: How to config the parameters to support higher concurrency for deploying the qwen2-7b model as an API at 8-GPU A800 (80G) server?

**Link**: https://github.com/vllm-project/vllm/issues/7325
**State**: closed
**Created**: 2024-08-09T02:39:50+00:00
**Closed**: 2024-08-13T03:08:08+00:00
**Comments**: 4
**Labels**: usage

### Description

### Your current environment

```text
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.2
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-78-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA A800-SXM4-80GB
GPU 1: NVIDIA A800-SXM4-80GB
GPU 2: NVIDIA A800-SXM4-80GB
GPU 3: NVIDIA A800-SXM4-80GB
GPU 4: NVIDIA A800-SXM4-80GB
GPU 5: NVIDIA A800-SXM4-80GB
GPU 6: NVIDIA A800-SXM4-80GB
GPU 7: NVIDIA A800-SXM4-80GB

Nvidia driver version: 550.90.07
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK availa

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Any plan run deepseek-r1 fp8 on Ampere gpu

**Link**: https://github.com/vllm-project/vllm/issues/13885
**State**: closed
**Created**: 2025-02-26T09:30:08+00:00
**Closed**: 2025-03-03T01:56:34+00:00
**Comments**: 1
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

I notice the core reason of vllm can not deployment deepseek-r1 fp8 model is `Marlin doesn't support block-wise fp8`. So, What are the possible approaches to run the DeepSeek R1 FP8 on Ampere architecture GPU

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: How to use breakpoints with VLLM to debug

**Link**: https://github.com/vllm-project/vllm/issues/13120
**State**: closed
**Created**: 2025-02-11T23:48:52+00:00
**Closed**: 2025-02-17T09:55:12+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment

None

### How would you like to use vllm

It seems like the execution of vllm is managed through a compiled engine, which makes it challenging to directly debug within an IDE using breakpoints. When debugging some custom funcs to the vllm, is there any way that we could disable compilation and just set breakpoints like normal python scripts? Thank you for your help!


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: The reasoning_parser doesn't dynamically apply to parser.add_argument after I added a custom Reasoning Parser

**Link**: https://github.com/vllm-project/vllm/issues/15999
**State**: closed
**Created**: 2025-04-03T07:42:16+00:00
**Closed**: 2025-04-10T06:55:42+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>
pip install vllm==0.8.2

```text

```

</details>


### 🐛 Describe the bug

Hello

I have declared a custom reasoning parser class and am preparing for online serving. However, the functions within this package alone cannot initialize the custom reasoning parser class class. 
Additionally, during xgrammars initialization, only deepseek_r1 is being used as the reasoner. 
When a custom reasoning parser class is created, I would like to be able to dynamically initialize the reasoner as well during the ReasoningParserManager in vllm engine initialization.

Thank you for your time and consideration.
Regard

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: ngc24.05 "RuntimeError: Cannot re-initialize CUDA in forked subprocess."

**Link**: https://github.com/vllm-project/vllm/issues/7246
**State**: closed
**Created**: 2024-08-07T04:45:15+00:00
**Closed**: 2024-08-13T22:40:18+00:00
**Comments**: 10
**Labels**: bug

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.2
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-3.10.0-1062.9.1.el7.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA A30
GPU 1: NVIDIA A30
GPU 2: NVIDIA A30
GPU 3: NVIDIA A30

Nvidia driver version: 525.85.12
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_engin

[... truncated for brevity ...]

---

## Issue #N/A: Offline Batched Inference with lora?

**Link**: https://github.com/vllm-project/vllm/issues/3001
**State**: closed
**Created**: 2024-02-23T02:15:48+00:00
**Closed**: 2024-02-27T21:57:14+00:00
**Comments**: 2

### Description

No description provided.

---

## Issue #N/A: [Bug]: Stuck at "generating GPU P2P access cache"

**Link**: https://github.com/vllm-project/vllm/issues/6893
**State**: closed
**Created**: 2024-07-29T08:35:14+00:00
**Closed**: 2024-08-04T18:31:53+00:00
**Comments**: 9
**Labels**: bug

### Description

### Your current environment

```text
PyTorch version: 2.3.1
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.1
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-113-generic-x86_64-with-glibc2.35
Is CUDA available: True

CUDA runtime version: 12.1.66
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA H100 PCIe
GPU 1: NVIDIA H100 PCIe

Nvidia driver version: 535.161.07
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      46 bits physical, 57 bits virtual
Byte Order:     

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: ValueError: not enough values to unpack (expected 22, got 21) when deploying DeepSeekV3

**Link**: https://github.com/vllm-project/vllm/issues/15453
**State**: closed
**Created**: 2025-03-25T09:12:15+00:00
**Closed**: 2025-03-26T09:33:45+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

For the environment, i simply use the vllm v0.8.0 docker image.

### 🐛 Describe the bug

When deploying DeepSeekV3 with TP=16 on two nodes, I encounter the `ValueError: not enough values to unpack (expected 22, got 21) ` error.

To create containers and ray, I use the  following commands, in which the `run_cluster.sh` refers to [vllm sample](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/run_cluster.sh).
```bash
# head node
DOCKER_IMAGE="xxx"
HEAD_NODE_ADDRESS="xxx"
NODE_TYPE="--head"  # Should be --head or --worker
PATH_TO_HF_HOME="xxx"

CRT_NODE_ADDRESS=${HEAD_NODE_ADDRESS}

bash run_cluster.sh \
        ${DOCKER_IMAGE} \
        ${HEAD_NODE_ADDRESS} \
        ${NODE_TYPE} \
        ${PATH_TO_HF_HOME} \
        -e VLLM_HOST_IP=${CRT_NODE_ADDRESS}

# other node
DOCKER_IMAGE="xxx"
HEAD_NODE_ADDRESS="xxx"
NODE_TYPE="--worker"  # Should be --head or --worker
PATH_TO_HF_HOME="xxx"

CRT_NODE_ADDRESS="xxx"

bash run_cluster.sh \
        

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Support for image linebreak tokens for vision model

**Link**: https://github.com/vllm-project/vllm/issues/17127
**State**: closed
**Created**: 2025-04-24T17:19:10+00:00
**Closed**: 2025-04-28T20:37:05+00:00
**Comments**: 4
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Currently when GPUModelRunner computes start and end indices https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py#L931-L934

it assumes the relative position of placeholder tokens is the same as `encoder_output`, however if we use image linebreak tokens this assumption no longer holds, which will lead to errors when chunked prefill is enabled.

For example, say an input image is split into 3 crops and each crop corresponds to 4 placeholder tokens, in our use case we add an <|im_linbreak|> token to separate each crop, the full image tokens would be
```
<|im_start|>
<|im_placeholder|><|im_placeholder|><|im_placeholder|><|im_placeholder|><|im_linebreak|>
<|im_placeholder|><|im_placeholder|><|im_placeholder|><|im_placeholder|><|im_linebreak|>
<|im_placeholder|><|im_placeholder|><|im_placeholder|><|im_placeholder|><|im_linebreak|>
<|im_end|>
```
the number of tokens between <|im_start|> and <|im_end|> is 15, however the l

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: V1 Engine crashes when sending requests with same request id

**Link**: https://github.com/vllm-project/vllm/issues/15041
**State**: closed
**Created**: 2025-03-18T14:49:35+00:00
**Closed**: 2025-03-20T17:01:03+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.12.0 | packaged by Anaconda, Inc. | (main, Oct  2 2023, 17:29:18) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-133-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.3.107
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4090
Nvidia driver version: 550.54.14
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

## Issue #N/A: [Feature]: Small Model Large Latency Compared to SGLang and TensorRT-LLM

**Link**: https://github.com/vllm-project/vllm/issues/7339
**State**: closed
**Created**: 2024-08-09T05:53:18+00:00
**Closed**: 2024-08-11T22:53:24+00:00
**Comments**: 2
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

In this post, https://lmsys.org/blog/2024-07-25-sglang-llama3/, it looks like vllm is not efficient in small model size in both online and offline benchmark. What is the bottleneck for vllm for small model inference and whether this will be addressed to catch SGLang and TensorRT performance.

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [Usage]: How to modify the default system prompt？

**Link**: https://github.com/vllm-project/vllm/issues/4497
**State**: closed
**Created**: 2024-04-30T09:23:20+00:00
**Closed**: 2024-05-07T05:13:08+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment

```text
vllm-0.4.0
```



### How would you like to use vllm

When I deploy an OpenAI-Compatible Server using `python -m vllm.entrypoints.openai.api_server`, how do I modify the default system prompt(You are a helpful assistant.)?



---

## Issue #N/A: [Usage]: Getting empty text using llm.generate of mixtral-8X7b-Instruct AWQ model

**Link**: https://github.com/vllm-project/vllm/issues/7375
**State**: closed
**Created**: 2024-08-09T18:28:26+00:00
**Closed**: 2024-08-11T23:38:29+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

I want to run inference of a Mixtral-8x7B-Instruct-v0.1-AWQ . Its giving me empty text as generation.
I am using the sample code from the model card:?
from vllm import LLM, SamplingParams
prompts = [
   "Tell me about AI",
   "Write a story about llamas",
   "What is 291 - 150?",
   "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
]
prompt_template=f'''[INST] {prompt} [/INST]'''

prompts = [prompt_template.format(prompt=prompt) for prompt in prompts]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ", quantization="awq", tensor_parallel_size=4)
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {ge

[... truncated for brevity ...]

---

## Issue #N/A: snapshot download from HF not from modelscope

**Link**: https://github.com/vllm-project/vllm/issues/2353
**State**: closed
**Created**: 2024-01-05T13:09:56+00:00
**Closed**: 2024-01-06T18:24:12+00:00
**Comments**: 5

### Description

since VLLM uses snapshot download from modelscope will not allow us to download private models , if it is implemented with snapshot download from HF it would be highly appreciable 

---

## Issue #N/A: [Bug]: Can't deserialize object reported by ray, H800*16 DeepSeek R1

**Link**: https://github.com/vllm-project/vllm/issues/15199
**State**: closed
**Created**: 2025-03-20T08:50:09+00:00
**Closed**: 2025-03-22T05:25:45+00:00
**Comments**: 4
**Labels**: bug, ray

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.12.9 (main, Mar 17 2025, 21:01:58) [Clang 20.1.0 ] (64-bit runtime)
Python platform: Linux-5.15.0-47-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA H800
GPU 1: NVIDIA H800
GPU 2: NVIDIA H800
GPU 3: NVIDIA H800
GPU 4: NVIDIA H800
GPU 5: NVIDIA H800
GPU 6: NVIDIA H800
GPU 7: NVIDIA H800

Nvidia driver version: 535.161.08
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True


[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Difference in language model usage post updating versions form 0.2 to 0.4 

**Link**: https://github.com/vllm-project/vllm/issues/4588
**State**: closed
**Created**: 2024-05-03T18:18:16+00:00
**Closed**: 2024-05-10T01:07:50+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment


```text
The output of `python collect_env.py`

Collecting environment information...
PyTorch version: 2.2.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Amazon Linux 2 (x86_64)
GCC version: (GCC) 7.3.1 20180712 (Red Hat 7.3.1-17)
Clang version: Could not collect
CMake version: version 3.27.7
Libc version: glibc-2.26

Python version: 3.10.9 | packaged by conda-forge | (main, Feb  2 2023, 20:20:04) [GCC 11.3.0] (64-bit runtime)
Python platform: Linux-5.10.213-201.855.amzn2.x86_64-x86_64-with-glibc2.26
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A10G
Nvidia driver version: 550.54.14
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Litt

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Free GPU memory upon destructing the Python client

**Link**: https://github.com/vllm-project/vllm/issues/16543
**State**: closed
**Created**: 2025-04-12T22:45:50+00:00
**Closed**: 2025-04-16T09:12:35+00:00
**Comments**: 2
**Labels**: usage

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.2 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.39

Python version: 3.12.7 (main, Apr  9 2025, 11:35:32) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-6.11.0-19-generic-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 12.8.93
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

Nvidia driver version: 570.124.06
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.8.0
/usr/lib/x86_64

[... truncated for brevity ...]

---

## Issue #N/A: FlashAttentionBackend only supports head sizes supported by xformers

**Link**: https://github.com/vllm-project/vllm/issues/3359
**State**: closed
**Created**: 2024-03-12T18:08:27+00:00
**Closed**: 2024-03-13T22:33:37+00:00
**Comments**: 3

### Description

`FlashAttentionBackend` currently only supports head sizes supported by `XFormersBackend`, specifically `[64, 80, 96, 112, 128, 256]`. Is there any reason to only support these head sizes with flash attention? If not, I can open a PR to remove this constraint (flash should support all dimensions up to 256) so that smaller models or those with unsupported head sizes can be used with vLLM w/flash attention.

```python
suppored_head_sizes = PagedAttentionImpl.get_supported_head_sizes()
if head_size not in suppored_head_sizes:
    raise ValueError(
        f"Head size {head_size} is not supported by PagedAttention. "
        f"Supported head sizes are: {suppored_head_sizes}.")
```

---

## Issue #N/A: [Bug]: Error while deserializing header: InvalidHeaderDeserialization

**Link**: https://github.com/vllm-project/vllm/issues/14596
**State**: closed
**Created**: 2025-03-11T04:34:08+00:00
**Closed**: 2025-03-13T04:05:37+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

root@node37:/disk1/qwen-2.5-vl-72b-in-awq-0226# docker compose -f docker-compose.yml down
[+] Running 2/2
 ✔ Container qwen-2.5-vl-72b-in-awq            Removed                                                                                                                                     2.0s 
 ✔ Network qwen-25-vl-72b-in-awq-0226_default  Removed                                                                                                                                     0.2s 
root@node37:/disk1/qwen-2.5-vl-72b-in-awq-0226# docker compose -f docker-compose.yml up -d
[+] Running 2/2
 ✔ Network qwen-25-vl-72b-in-awq-0226_default  Created                                                                                                                                     0.1s 
 ✔ Container qwen-2.5-vl-72b-in-awq            Started                                                                                                                            

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Multi lora inference support for llava v1.6

**Link**: https://github.com/vllm-project/vllm/issues/13034
**State**: closed
**Created**: 2025-02-10T11:27:11+00:00
**Closed**: 2025-02-17T10:09:54+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

I tried to run multi-lora inference on vllm for llava-hf/llava-v1.6-mistral-7b-hf. But i found it not supported according to the doc here https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-multimodal-language-models. 

Is there any plan to support multi lora for more mllm in near future or any practical way to obtain it on llava-next?

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Installation]: git clone cutlass fails

**Link**: https://github.com/vllm-project/vllm/issues/7368
**State**: closed
**Created**: 2024-08-09T15:29:52+00:00
**Closed**: 2024-08-11T22:54:41+00:00
**Comments**: 7
**Labels**: installation

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Red Hat Enterprise Linux release 8.10 (Ootpa) (x86_64)
GCC version: (GCC) 8.5.0 20210514 (Red Hat 8.5.0-22)
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.28

Python version: 3.11.9 (main, Jun 19 2024, 10:02:06) [GCC 8.5.0 20210514 (Red Hat 8.5.0-22)] (64-bit runtime)
Python platform: Linux-4.18.0-553.8.1.el8_10.x86_64-x86_64-with-glibc2.28
Is CUDA available: True
CUDA runtime version: 12.2.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA L40S-48C
GPU 1: NVIDIA L40S-48C
GPU 2: NVIDIA L40S-48C

Nvidia driver version: 535.129.03
cuDNN version: Probably one of the following:
/usr/lib64/libcudnn.so.9.3.0
/usr/lib64/libcudnn_adv.so.9.3.0
/usr/lib64/libcudnn_cnn.so.9.3.0
/usr/lib64/libcudnn_engines_

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: HIP error: invalid device function

**Link**: https://github.com/vllm-project/vllm/issues/17170
**State**: closed
**Created**: 2025-04-25T08:13:42+00:00
**Closed**: 2025-04-30T02:03:34+00:00
**Comments**: 5
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

INFO 04-25 08:07:32 [__init__.py:239] Automatically detected platform rocm.
Collecting environment information...
PyTorch version: 2.7.0a0+git1341794
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.4.43482-0f2d60242

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.0 25133 c7fe45cf4b819c5991fe208aaa96edf142730f1d)
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.12.10 (main, Apr  9 2025, 08:55:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-57-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI100 (gfx908:sramecc+:xnack-)
Nvidia driver version: Could not collect
cuDNN version: Cou

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: error when set  VLLM_ATTENTION_BACKEND=FLASHINFER

**Link**: https://github.com/vllm-project/vllm/issues/13258
**State**: closed
**Created**: 2025-02-14T03:52:30+00:00
**Closed**: 2025-02-17T00:31:59+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

<details>

```text
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.11.9 | packaged by conda-forge | (main, Apr 19 2024, 18:36:13) [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-5.15.0-131-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.5.82
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

Nvidia driver version: 550.127.08
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.2.1
/usr/lib/x

[... truncated for brevity ...]

---

