# await-response - issues

**Total Issues**: 28
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 2
- Closed Issues: 26

### Label Distribution

- await-response: 28 issues
- inactive: 4 issues
- unable-reproduce: 2 issues
- amd: 2 issues
- bug: 2 issues
- flashinfer: 1 issues

---

## Issue #N/A: [Feature] does sglang can be installed based on a pre-installed pytorch ?

**Link**: https://github.com/sgl-project/sglang/issues/6534
**State**: open
**Created**: 2025-05-22T18:09:28+00:00
**Comments**: 1
**Labels**: await-response

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

We're developing some features that needs higher version of torch and nvidia-cuda-runtime
just like vLLM's 
https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-from-source

![Image](https://github.com/user-attachments/assets/7c26f9c7-ca03-4f2b-8023-cf99d4560a46)

### Related resources

_No response_

---

## Issue #N/A: [Feature] support calculate-kv-scales for fp8 kvcache

**Link**: https://github.com/sgl-project/sglang/issues/6518
**State**: open
**Created**: 2025-05-22T05:42:02+00:00
**Comments**: 1
**Labels**: await-response

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Do we support --calculate-kv-scales like vLLM does for kv-cache quant?

If we use --kv-cache-dtype fp8_e4m3, the scale factor is 1.0, how can we calculate kv scales in runtime like vLLM.

### Related resources

_No response_

---

## Issue #N/A: [Benchmark] sglang successful requests issue (may related to env)

**Link**: https://github.com/sgl-project/sglang/issues/2805
**State**: closed
**Created**: 2025-01-09T06:20:36+00:00
**Closed**: 2025-03-11T00:17:36+00:00
**Comments**: 3
**Labels**: await-response, inactive

### Description

sglang，0.4.0.post2

python -m sglang.launch_server --model-path /mnt/home/Llama-3.1-8B-Instruct --enable-torch-compile --disable-radix-cache

python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --dataset-path /mnt/home/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 3000 --output-file /mnt/home/offline_sglang.jsonl
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max reqeuest concurrency:                not set   
Successful requests:                     1648      
Benchmark duration (s):                  170.41    
Total input tokens:                      369103    
Total generated tokens:                  326408    
Total generated tokens (retokenized):    326356    
Request throughput (req/s):              9.67      
Input token throughput (tok/s):          2165.94   
Output token throughput (tok/s):         1915.40   
Total token th

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] tp == 2 model gibberish

**Link**: https://github.com/sgl-project/sglang/issues/2354
**State**: closed
**Created**: 2024-12-04T22:38:41+00:00
**Closed**: 2025-01-05T04:49:50+00:00
**Comments**: 2
**Labels**: await-response, unable-reproduce

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

I've been having issues with tensor parallelism tp=2 on various Llama models. The model outputs gibberish with tp=2 but performs fine without it.

### Reproduction

#### With tensor parallelism
Terminal 1
```python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000 --host 0.0.0.0 --tp 2```


[... truncated for brevity ...]

---

## Issue #N/A: [Bug] amdgpu，tp-size=2，Detected errors during sampling! NaN in the logits.

**Link**: https://github.com/sgl-project/sglang/issues/1953
**State**: closed
**Created**: 2024-11-08T04:44:58+00:00
**Closed**: 2025-01-29T00:16:25+00:00
**Comments**: 6
**Labels**: await-response, inactive, amd

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

python3 -m sglang.launch_server --model-path  /root/.xinference/cache/qwen2_5-instruct-gptq-7b-Int8/ --port 30000 --mem-fraction-static  0.8 --kv-cache-dtype int8 --attention-backend triton --sampling-backend pytorch --tp-size 2
WARNING 11-08 04:42:43 rocm.py:13] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is over

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] tp-size=2，model launch error

**Link**: https://github.com/sgl-project/sglang/issues/1945
**State**: closed
**Created**: 2024-11-07T06:14:03+00:00
**Closed**: 2025-01-29T00:16:26+00:00
**Comments**: 5
**Labels**: await-response, inactive, amd

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

tp-size=2, model launch is frozen.

### Reproduction

 python3 -m sglang.launch_server --model-path  /root/.xinference/cache/qwen2_5-instruct-gptq-7b-Int8/ --port 30000 --mem-fraction-static  0.8 --tp-size 2 --kv-cache-dtype int8 --attention-backend triton --sampling-backend pytorch --enable-torch-compile

### Environment

amd gpu RTX 7900

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] subprocess.CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmpx4yubctp/main.c', '-O3', '-shared', '-fPIC', '-o', '/tmp/tmpx4yubctp/cuda_utils.cpython-310-x86_64-linux-gnu.so', '-lcuda', '-L/home/adminad/anaconda3/envs/py10/lib/python3.10/site-packages/triton/backends/nvidia/lib'

**Link**: https://github.com/sgl-project/sglang/issues/1240
**State**: closed
**Created**: 2024-08-28T09:46:50+00:00
**Closed**: 2024-09-22T12:52:08+00:00
**Comments**: 2
**Labels**: await-response

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

Traceback (most recent call last):
  File "/home/adminad/anaconda3/envs/py10/lib/python3.10/site-packages/sglang/srt/managers/tp_worker.py", line 878, in run_tp_server
    model_server.exposed_step(recv_reqs)
  File "/home/adminad/anaconda3/envs/py10/lib/python3.10/site-packages/sglang/srt/managers/tp_worker.py", line 234, in exposed_st

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] add option to use liger triton kernel

**Link**: https://github.com/sgl-project/sglang/issues/1216
**State**: closed
**Created**: 2024-08-26T01:28:13+00:00
**Closed**: 2024-09-01T00:50:49+00:00
**Comments**: 5
**Labels**: await-response

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

liger triton kernel is a one liner patch to huggingface models. It provides inference speed up and memory reduction.


### Related resources

https://github.com/linkedin/Liger-Kernel

---

## Issue #N/A: Accuracy degrading in concurrent scenario

**Link**: https://github.com/sgl-project/sglang/issues/1203
**State**: closed
**Created**: 2024-08-25T03:00:16+00:00
**Closed**: 2024-09-22T12:51:30+00:00
**Comments**: 2
**Labels**: await-response

### Description

Hi, I have tested that when the concurrency is 1, the accuracy is expected. However, when concurrency increases, accuracy degrades. I have checked that no decoding oom happened. From the log, there also seems to have no exception.

The model is qwen2-7b-awq.

---

## Issue #N/A: [Feature] add disable_custom_all_reduce

**Link**: https://github.com/sgl-project/sglang/issues/1118
**State**: closed
**Created**: 2024-08-16T04:59:48+00:00
**Closed**: 2024-08-21T04:53:40+00:00
**Comments**: 7
**Labels**: await-response

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Sometimes, we need to turn off Custom allreduce. 
Please  support disable_custom_all_reduce.


### Related resources

_No response_

---

## Issue #N/A: [Bug] After service, `torch.distributed.DistBackendError`

**Link**: https://github.com/sgl-project/sglang/issues/1116
**State**: closed
**Created**: 2024-08-16T03:27:38+00:00
**Closed**: 2024-09-22T12:44:54+00:00
**Comments**: 2
**Labels**: await-response

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

I have 4 A100(40G), I followed the instructions below:
`python -m sglang.launch_server --model-path /ldata/llms/Meta-Llama-3.1-70B-Instruct --host 0.0.0.0 --port 30000 --tp 4 --mem-fraction-static 0.95 --disable-cuda-graph`
Then it reported an error:
```
torch.distributed.DistBackendError: NCCL error in: ../torch/csrc/distrib

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Can't run Qwen2-57B-A14B-Instruct-GPTQ-Int4

**Link**: https://github.com/sgl-project/sglang/issues/1100
**State**: closed
**Created**: 2024-08-14T12:39:32+00:00
**Closed**: 2024-09-22T12:41:49+00:00
**Comments**: 4
**Labels**: await-response

### Description

### Describe the bug

I can't start sglang with model qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4, below is the error ouput.
Does sglang support it now ?

python -m sglang.launch_server --quantization gptq --model-path qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4 --port 8000 --disable-flashinfer-sampling --disable-flashinfer --tp 2 --enable-p2p-check

server_args=ServerArgs(model_path='qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4', tokenizer_path='qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4', tokenizer_mode='auto', skip_tokenizer_init=False, load_format='auto', dtype='auto', trust_remote_code=False, context_length=None, quantization='gptq', served_model_name='qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4', chat_template=None, host='127.0.0.1', port=8000, additional_ports=[8001, 8002, 8003, 8004], mem_fraction_static=0.87, max_running_requests=None, max_num_reqs=None, max_total_tokens=None, chunked_prefill_size=None, max_prefill_tokens=16384, schedule_policy='lpm', schedule_conservativeness=1.0, tp_size=2, s

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Always Watch Dog TimeOut

**Link**: https://github.com/sgl-project/sglang/issues/1093
**State**: closed
**Created**: 2024-08-14T09:55:55+00:00
**Closed**: 2024-09-23T03:55:28+00:00
**Comments**: 5
**Labels**: await-response, unable-reproduce

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

### Describe the bug

I frequently encounter Watch Dog TimeOut errors when deploying Mistral-123B using 8x A800 80G, which causes the service to stop. This issue occurs whether I send a single request or multiple requests. Below are my startup command and logs.

Command：
python -m sglang.launch_server --model-path /Mistral-Large-Instruct-2/ --host 0.0.0.0 --port 9997 --disable-cuda-graph --schedule-conservativeness

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] cuda out of memory when using MQA and input_len=output_len=1024

**Link**: https://github.com/sgl-project/sglang/issues/1087
**State**: closed
**Created**: 2024-08-14T04:25:52+00:00
**Closed**: 2024-09-22T13:02:56+00:00
**Comments**: 4
**Labels**: await-response

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

### Describe the bug

We have pretrained a 7B model using MQA(num_key_value_heads=1), when I do throughput benchmarking by modifying the config of meta-llama-3, setting num_key_value_heads=1. The service collapse when receiving workloads with input_len=output_len=1024.

### Reproduction

serving:
`python3 -m sglang.launch_server --model-path /models/dummy --disable-radix-cache`
where `/models/dummy` is simply copied

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Are there plans to implement a prefill-decode split inference architecture?

**Link**: https://github.com/sgl-project/sglang/issues/1080
**State**: closed
**Created**: 2024-08-13T14:18:46+00:00
**Closed**: 2024-09-22T14:23:45+00:00
**Comments**: 2
**Labels**: await-response

### Description

### Motivation
Related work includes:

1.https://github.com/LLMServe/DistServe/tree/main
2.https://github.com/vllm-project/vllm/pull/2809
3.mooncake has proven that separating prefill and decode can lead to throughput improvements and significant cost savings for online services. Are there any plans to do this?

### Related resources

_No response_

---

## Issue #N/A: [Feature] Inference speed difference between sglang and vllm is smaller than advertised

**Link**: https://github.com/sgl-project/sglang/issues/998
**State**: closed
**Created**: 2024-08-09T08:07:39+00:00
**Closed**: 2024-08-15T16:26:10+00:00
**Comments**: 8
**Labels**: await-response

### Description

### Motivation

I compared the inference speed of two large model inference frameworks. I found that sglang is only about 30% faster than vllm, which is much lower than the claimed 3.8 times speedup.

Below are my environment details, prompt, and inference results.
my environment：
gpu:4090*1
cuda:12.4
Python:3.11.0
vllm:0.5.3
sglang:0.2.7

launch command:
`python -m vllm.entrypoints.openai.api_server --model /home/modeldata/Qwen2-1.5B-Instruct --port 8899`
`python -m sglang.launch_server --model-path /home/modeldata/Qwen2-1.5B-Instruct --host 0.0.0.0 --port 30000`

prompt:
`请根据用户反馈，仔细思考标准答案构成要素，并改写出5句答案\n你是直播真人问答客服，为避免客服回答的答案重复度过高，请你逐句思考并改写问题的答案。\n****************\n#样例\n用户问题：声音好好听\n参考的问答对：["问题: 声音好好听, 答案: 谢谢宝宝的夸奖，喜欢主播的可以点个关注", "问题: 你是真人吗, 答案: 什么，你说我是不是真人", "问题: 没有红包吗, 答案: 红包左上角都会安排的", "问题: 拍啦, 答案: 好的感谢支持咱家玉米"]\n输出格式：["感谢你的夸赞支持", "你的夸赞是我前进的动力", "收到你的夸奖，心情美美哒", "夸奖收到，谢谢宝宝的热情", "你的夸奖我收到了，谢谢"]\n\n****************\n#规则（必须严格遵循）\n1、你的答案必须仿照改写用户觉得满意的答案\n2、你的答案绝对不能按照用户不满意答案的写法。

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] requires "python-multipart" to be installed with docker image

**Link**: https://github.com/sgl-project/sglang/issues/949
**State**: closed
**Created**: 2024-08-06T08:03:23+00:00
**Closed**: 2024-08-16T08:09:06+00:00
**Comments**: 18
**Labels**: await-response, flashinfer

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

I tested sglang in Kubernetes with the minimum configuration below:
```
  containers:
  - args:
    - --model-path
    - /workspace/models/models--facebook--opt-125m
    - --served-model-name
    - opt-125m
    - --host
    - 0.0.0.0
    - --port
    - "8080"
    command:
    - python3
    - -m
    - sglang.launch_server
    image: lmsysorg/sglang:v0.2.9-cu121
```

However, it emits error like:
```
Form data requires "python-multipart" to be installed.
You can install "python-multipart" with:

pip install python-multipart

Traceback (most re

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] disable_flashinfer didn't take effect

**Link**: https://github.com/sgl-project/sglang/issues/945
**State**: closed
**Created**: 2024-08-06T06:46:53+00:00
**Closed**: 2024-08-06T09:35:48+00:00
**Comments**: 10
**Labels**: await-response

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

I tried pip install flashinfer, but got error from
```
 File "/usr/local/python/lib/python3.10/site-packages/sglang/srt/model_executor/model_runner.py", line 28, in <module>
    from flashinfer import (
ModuleNotFoundError: No module named 'flashinfer'
```

According to readme, i tried to set disable_flashinfer when init the runtime, my code is like

```
self.runtime = sgl.Runtime(
            model_path = self.loader.target_path,
            tp_size = envs.TENSOR_PARALLEL_SIZE,
            trust_remote_code = True,
            max_num_reqs = 40,
            d

[... truncated for brevity ...]

---

## Issue #N/A: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method

**Link**: https://github.com/sgl-project/sglang/issues/944
**State**: closed
**Created**: 2024-08-06T02:57:12+00:00
**Closed**: 2024-08-08T09:27:10+00:00
**Comments**: 2
**Labels**: await-response

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server --model-path /mnt/afs/data/model/open_source_data/Qwen/Qwen2-7B-Instruct --port 30000
server_args=ServerArgs(model_path='/mnt/afs/data/model/open_source_data/Qwen/Qwen2-7B-Instruct', tokenizer_path='/mnt/afs/data/model/open_source_data/Qwen/Qwen2-7B-Instruct', tokenizer_mode='auto', load_format='auto', dtype='auto', trust_remote_code=False, context_length=None, quantization=None, served_model_name='/mnt/afs/data/model/open_source_data/Qwen/Qwen2-7B-Instruct', chat_template=None, host='127.0.0.1', port=30000, additional_p

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Fail to build from Docker

**Link**: https://github.com/sgl-project/sglang/issues/885
**State**: closed
**Created**: 2024-08-02T06:10:22+00:00
**Closed**: 2024-08-06T09:47:09+00:00
**Comments**: 3
**Labels**: await-response

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

Building from the Dockerfile, got gpg-key error like:

```
Err:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
  The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
```

while executing:

```
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common \
    && add-apt-reposito

[... truncated for brevity ...]

---

## Issue #N/A: RuntimeError: TopKTopPSamplingFromProbs failed with error code no kernel image is available for execution on the device  已杀死[Bug] 

**Link**: https://github.com/sgl-project/sglang/issues/865
**State**: closed
**Created**: 2024-08-01T09:10:16+00:00
**Closed**: 2024-09-22T13:05:44+00:00
**Comments**: 4
**Labels**: bug, await-response

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

RuntimeError: TopKTopPSamplingFromProbs failed with error code no kernel image is available for execution on the device

已杀死

### Reproduction

RuntimeError: TopKTopPSamplingFromProbs failed with error code no kernel image is available for execution on the device

已杀死

### Environment

```Shell
RuntimeError: TopKTopPSamplingFromProbs failed with error code no kernel image is available for execution on the device

已杀死
```


---

## Issue #N/A: [Bug] OOM when benchmarking 

**Link**: https://github.com/sgl-project/sglang/issues/810
**State**: closed
**Created**: 2024-07-29T22:26:35+00:00
**Closed**: 2024-07-30T08:59:02+00:00
**Comments**: 6
**Labels**: await-response

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

out of memory encountered

`Exception in ModelTpServer:
Traceback (most recent call last):
  File "/opt/tiger/sglang/python/sglang/srt/managers/controller/tp_worker.py", line 209, in exposed_step
    self.forward_step()
  File "/home/tiger/.local/lib/python3.9/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/opt/tiger/sglang/python/sglang/srt/managers/controller/tp_worker.py", line 240, in forward_step
    self.forward_decode_batch(self.running_batch)
  File "/opt/tiger/sglang/python/sglang/srt/manag

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]Why is the qwen2 openai interface accessing the streaming output incomplete 

**Link**: https://github.com/sgl-project/sglang/issues/765
**State**: closed
**Created**: 2024-07-27T06:41:06+00:00
**Closed**: 2024-07-27T11:20:23+00:00
**Comments**: 5
**Labels**: await-response

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug



```
from openai import OpenAI
# from openai._client import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://192****:30000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
## 流式回答
stream = client.chat.completions.create(
model="/ai/qwen2-7b",

messages=[{"role": "user", "content": "介绍广州"}],
stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        # print(chunk.choices[0].delta.content)
        print(chunk.choice

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Can't run server on sm75

**Link**: https://github.com/sgl-project/sglang/issues/748
**State**: closed
**Created**: 2024-07-26T13:55:45+00:00
**Closed**: 2024-07-31T07:52:23+00:00
**Comments**: 14
**Labels**: bug, await-response

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

Qwen2-72B-Instruct-GPTQ-Int4
Can't run this server

### Reproduction

root@4563caaa7539:/sgl-workspace# CUDA_VISIBLE_DEVICES=9,0,2,4 python3 -m sglang.launch_server  --model-path /root/hf_model/Qwen/Qwen2-72B-Instruct-GPTQ-Int4   --dtype half    --trust-remote-code   --tp-size 4     --quantization gptq     --log-level INFO        --enable-p2p-check      --efficient-weight-load         --host 0.0.0.0  --log-requests  --show-time-cost      --disable-disk-cache    --enable-torch-compile  --mem-fraction-static 0.6       --disable-cuda-graph    --max-running-requests 64       --

[... truncated for brevity ...]

---

## Issue #N/A: Initialization failed. warmup error: 

**Link**: https://github.com/sgl-project/sglang/issues/744
**State**: closed
**Created**: 2024-07-26T11:49:45+00:00
**Closed**: 2024-11-14T19:27:06+00:00
**Comments**: 9
**Labels**: await-response

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

I execute :python -m sglang.launch_server --model-path Qwen/Qwen2-7B-Instruct  --tp 2 --mem-fraction-static 0.7  --enable-p2p-check --host 0.0.0.0 --port 8000    follow error:
[gpu_id=0] Load weight end. type=Qwen2ForCausalLM, dtype=torch.bfloat16, avail mem=15.61 GB
[gpu_id=1] Load weight end. type=Qwen2ForCausalLM, dtype=torch.bfloat16, avail mem=15.95 GB
[gpu_id=0] Memory pool end. avail mem=6.12 GB
[gpu_id=1] Memory pool end. avail mem=6.46 GB
[gpu_id=1] Capture cuda graph begin. This can take up to several minutes.
[gpu_id=0] Capture cuda graph begin. This

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] process not terminated after PM2 is kill

**Link**: https://github.com/sgl-project/sglang/issues/680
**State**: closed
**Created**: 2024-07-21T00:11:31+00:00
**Closed**: 2024-08-01T10:51:40+00:00
**Comments**: 4
**Labels**: await-response

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.

### Describe the bug

I use pm2 to run the server and it appears the python process is still running after the pm2 is killed, the GPUs were still occupied. How do I properly terminate the process?

### Reproduction

pm2 start /usr/bin/python --name sglang-launch-server -- -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000

### Environment

```Shell
N/A
```


---

## Issue #N/A: select() on first assistant token broken (in different ways in Mistral and Llama). Likely tokenization issue.

**Link**: https://github.com/sgl-project/sglang/issues/608
**State**: closed
**Created**: 2024-07-12T08:12:53+00:00
**Closed**: 2024-09-24T01:11:35+00:00
**Comments**: 4
**Labels**: await-response, inactive

### Description

Below is simple code with the output, showing that Llama and Mistral choose a clearly nonsensical first token in `select()` depending on whether the assistant message contains a leading space, but each in the opposite way.

```
@sglang.function
def selected(s):
    s += sglang.user('Hi!')
    s += sglang.assistant_begin()
    s += sglang.gen('var', choices=('Hello', 'Goodbye', '^*B^&A'))
    s += sglang.assistant_end()
print(selected.run(backend=runtime).text())
# Mistral: [INST] Hi! [/INST]^*B^&A </s><s>
#   LLAMA: [INST] Hi! [/INST]Hello </s><s>

@sglang.function
def selected_space(s):
    s += sglang.user('Hi!')
    s += sglang.assistant_begin()
    s += ' '
    s += sglang.gen('var', choices=('Hello', 'Goodbye', '^*B^&A'))
    s += sglang.assistant_end()
print(selected_space.run(backend=runtime).text())
# Mistral: [INST] Hi! [/INST] Hello </s><s>
#   LLAMA: [INST] Hi! [/INST] ^*B^&A </s><s>

@sglang.function
def freeform(s):
    s += sglang.user('Hi!')
  

[... truncated for brevity ...]

---

## Issue #N/A: Mistral v0.3 Weight Loading

**Link**: https://github.com/sgl-project/sglang/issues/519
**State**: closed
**Created**: 2024-06-09T10:33:27+00:00
**Closed**: 2024-07-30T03:39:11+00:00
**Comments**: 2
**Labels**: await-response

### Description

Failed to launch the sglang server:

Run command: `python -m sglang.launch_server --model-path Mistral-7B-Instruct-v0.3 --port 30000`

Error message: `KeyError: 'layers.0.attention.wk.weight'`

This issue is also [reported](https://github.com/vllm-project/vllm/issues/5061) and [fixed](https://github.com/vllm-project/vllm/pull/5005) in vLLM.

---

