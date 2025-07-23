# quant - issues

**Total Issues**: 18
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 18

### Label Distribution

- quant: 18 issues
- inactive: 7 issues
- high priority: 6 issues
- help wanted: 5 issues
- performance: 4 issues
- deepseek: 4 issues
- bug: 4 issues
- good first issue: 3 issues
- speculative-decoding: 1 issues
- enhancement: 1 issues

---

## Issue #N/A: [Feature] support DeepSeek R1 FP4

**Link**: https://github.com/sgl-project/sglang/issues/5055
**State**: closed
**Created**: 2025-04-04T01:11:06+00:00
**Closed**: 2025-06-04T00:19:47+00:00
**Comments**: 4
**Labels**: high priority, inactive, performance, quant, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled cc @Edwardf0t1 @kushanam @elfiegg 

Optimization is also important on Blackwell

### Related resources

_No response_

---

## Issue #N/A: [Bug] fix dsv3 awq issue

**Link**: https://github.com/sgl-project/sglang/issues/4462
**State**: closed
**Created**: 2025-03-16T05:27:20+00:00
**Closed**: 2025-04-07T02:17:41+00:00
**Comments**: 4
**Labels**: bug, high priority, performance, quant

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

as titled

### Reproduction

N/A

### Environment

N/A

---

## Issue #N/A: [Accuracy] [Online Quantization] Llama 1B FP16/FP8/W8A8_FP8 accuracy

**Link**: https://github.com/sgl-project/sglang/issues/4434
**State**: closed
**Created**: 2025-03-14T18:50:27+00:00
**Closed**: 2025-03-17T07:28:58+00:00
**Comments**: 2
**Labels**: bug, high priority, quant

### Description

## Conclusion
W8A8_FP8 quantization doesn't support online quantization


### GSM8K

#### Preparation
```bash
curl -o test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl

kubectl cp /Users/bhe/Desktop/oss/data/gsm8k/test.jsonl nfs_host:/shared/public/data/gsm8k/test.jsonl
```

FP16 Baseline:
```bash
python3 -m sglang.launch_server --model /shared/public/models/meta-llama/Llama-3.2-1B-Instruct --trust-remote-code

python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
100%|████████████████████████████████████| 1319/1319 [00:10<00:00, 121.39it/s]
Accuracy: 0.396
Invalid: 0.003
Latency: 10.905 s
Output throughput: 11035.006 token/s
```


FP8
```bash
python3 -m sglang.launch_server --model /shared/public/models/meta-llama/Llama-3.2-1B-Instruct --quantization fp8 --trust-remote-code


python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
100%|██████████████████

[... truncated for brevity ...]

---

## Issue #N/A: Speculative Decoding Fails with AWQ Quantized Model

**Link**: https://github.com/sgl-project/sglang/issues/4351
**State**: closed
**Created**: 2025-03-12T21:30:32+00:00
**Closed**: 2025-05-13T00:19:02+00:00
**Comments**: 2
**Labels**: inactive, quant, speculative-decoding

### Description

Description:

I am facing an issue when using speculative decoding with an AWQ quantized model in sglang. The same configuration works fine with an unquantized model (Llama-3.1-8b-Instruct), but fails when I switch to an AWQ quantized model(Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4).

Setup:

GPUs: NVIDIA L40s (48GB VRAM) x 2
CUDA Version: 12.8
PyTorch Version: 2.5.1
sglang Version: 0.4.3.post2

Working Configuration (Unquantized Model):

```
import requests
import os

from sglang import assistant_begin, assistant_end
from sglang import assistant, function, gen, system, user
from sglang import image
from sglang import RuntimeEndpoint, set_default_backend
from sglang.srt.utils import load_image
from sglang.test.test_utils import is_in_ci
from sglang.utils import print_highlight, terminate_process, wait_for_server
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if is_in_ci():
    from sglang.docs.frontend.patch import launch_server_cmd
else:
    from sglang.utils import launch_serve

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] fix DeepSeek V2/V3 awq

**Link**: https://github.com/sgl-project/sglang/issues/4338
**State**: closed
**Created**: 2025-03-12T09:23:32+00:00
**Closed**: 2025-04-08T06:26:05+00:00
**Comments**: 9
**Labels**: bug, good first issue, help wanted, quant

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to integrate the awq dequant from sgl-kernel and found that both the main version and the integrated version have issues with the awq of DeepSeek V2 Coder and DeepSeek V3, which need to be fixed.

```
casperhansen/deepseek-coder-v2-instruct-awq
cognitivecomputations/DeepSeek-V3-AWQ
```

### Reproduction

N/A

### Environment

N/A

---

## Issue #N/A: [Bug] fix gemma-2-2b-it-FP8 accuracy

**Link**: https://github.com/sgl-project/sglang/issues/4324
**State**: closed
**Created**: 2025-03-12T01:27:58+00:00
**Closed**: 2025-05-21T09:30:43+00:00
**Comments**: 8
**Labels**: bug, good first issue, help wanted, high priority, quant

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

The accuracy of `neuralmagic/gemma-2-2b-it-FP8` drops from 0.62 to 0.52 in the main branch. It was detected by our nightly CI run. We need to fix this.

```
neuralmagic/gemma-2-2b-it-FP8 | 0.512 | 0.6
```
https://github.com/sgl-project/sglang/actions/runs/13800885290

### Reproduction

N/A

### Environment

N/A

---

## Issue #N/A: [Bug] Accuracy issue with SGLang using DeepSeek-R1-AWQ

**Link**: https://github.com/sgl-project/sglang/issues/4158
**State**: closed
**Created**: 2025-03-07T04:04:05+00:00
**Closed**: 2025-04-07T06:40:34+00:00
**Comments**: 11
**Labels**: quant

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

The inference result of a simple question such as "9.11 and 9.8 which is greater?" can often time (~4 out of 5 times) result in progressively meaningless texts as more tokens are being generated.

The model checkpoint is: https://huggingface.co/cognitivecomputations/DeepSeek-R1-AWQ

SGlang installation from pip install "sglang[all]>=0.4.3.

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] cuda kernel illegal and  GPTQMarlinMoEMethod.apply() got an unexpected keyword argument 'correction_bias'

**Link**: https://github.com/sgl-project/sglang/issues/4083
**State**: closed
**Created**: 2025-03-05T06:47:58+00:00
**Closed**: 2025-03-16T07:43:21+00:00
**Comments**: 10
**Labels**: quant, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I test the model(OPEA/DeepSeek-R1-int4-gptq-sym-inc from HF) with the excellent sglang and two nodes(2 x 8X80G A100). 

COMMANDS:
**python3 -m sglang.launch_server --model-path /data/LM/hf/DeepSeek-R1-int4-gptq-sym-inc --tp 16 --dist-init-addr xx.yy.zz.210:25000 --nnodes 2 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 30000** 
an

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Config file not found when use NVIDIA_H20-3e

**Link**: https://github.com/sgl-project/sglang/issues/4028
**State**: closed
**Created**: 2025-03-03T12:12:14+00:00
**Closed**: 2025-03-05T03:29:07+00:00
**Comments**: 3
**Labels**: quant

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I use H20 141G, the device name is NVIDIA_H20-3e，sglang logs:

Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/quantization/configs/N=7168,K=2048,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json

Using default MoE 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] KeyError: 'model.layers.0.mlp.down_proj.weight_scale_inv' when run deepseek 671b with 64 RTX 4090 GPU

**Link**: https://github.com/sgl-project/sglang/issues/4018
**State**: closed
**Created**: 2025-03-03T09:03:05+00:00
**Closed**: 2025-05-22T00:19:08+00:00
**Comments**: 4
**Labels**: help wanted, inactive, quant, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

run deepseek r1 671b with 8 nodes,each nodes with 8 rtx4090 GPU,
then follow this problem delete [quantization_config](https://github.com/sgl-project/sglang/issues/3491#issuecomment-2650779851) 

![Image](https://github.com/user-attachments/assets/1990c4e7-cfac-4ef1-9d83-c8a27fb6c00a)

### Reproduction

run 8 nodes one by one:
export GLOO_

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] TypeError: AWQMoEMethod.create_weights() missing 1 required positional argument: 'intermediate_size_per_partition'

**Link**: https://github.com/sgl-project/sglang/issues/3476
**State**: closed
**Created**: 2025-02-11T01:29:30+00:00
**Closed**: 2025-02-12T01:48:29+00:00
**Comments**: 8
**Labels**: quant, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
[2025-02-10 13:35:43 TP3] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 1787, in run_scheduler_process
    scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank, dp_rank)
  File "/sgl-workspace/sglang/python/sglang/srt/managers/schedul

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] TypeError: _ColumnvLLMParameter.load_column_parallel_weight() got an unexpected keyword argument 'tp_rank'

**Link**: https://github.com/sgl-project/sglang/issues/3464
**State**: closed
**Created**: 2025-02-10T07:25:10+00:00
**Closed**: 2025-02-12T14:11:03+00:00
**Comments**: 9
**Labels**: help wanted, quant

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
Cache shape torch.Size([163840, 64])
Loading safetensors checkpoint shards:   0% Completed | 0/74 [00:00<?, ?it/s]
[2025-02-09 22:50:08 TP7] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 1787, in run_scheduler_process
    scheduler = Scheduler

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] TypeError: AWQMoEMethod.create_weights() missing 1 required positional argument: 'intermediate_size_per_partition' when running deepseek-r1-awq model

**Link**: https://github.com/sgl-project/sglang/issues/3303
**State**: closed
**Created**: 2025-02-05T00:45:19+00:00
**Closed**: 2025-04-13T00:43:14+00:00
**Comments**: 5
**Labels**: help wanted, inactive, quant

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When running deepseek-r1-awq model:
TypeError: AWQMoEMethod.create_weights() missing 1 required positional argument: 'intermediate_size_per_partition'

### Reproduction

python3 -m sglang.launch_server --model-path "cognitivecomputations/DeepSeek-R1-AWQ" --tp 8 --port 8411 --host 0.0.0.0 --context-length 8192 --trust-remote-code

### Envir

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] FP8 weight only w8a16 quantization native support

**Link**: https://github.com/sgl-project/sglang/issues/3007
**State**: closed
**Created**: 2025-01-20T10:50:31+00:00
**Closed**: 2025-03-23T00:19:17+00:00
**Comments**: 3
**Labels**: inactive, quant

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Hi,

I was using VLLM for inference and I am using A10 GPU which doesnt have w8a8 fp8 support. But when I use (without quantization beforehand)

`
./vllm_docker.sh meta-llama/Llama-3.1-8B-Instruct --quantization fp8
`

the server starts with 

> Your GPU does not have native support for FP8 computation but FP8 quantization is being used. Weight-only FP8 compression will be used leveraging the Marlin kernel. This may degrade performance for compute-heavy workloads.


I am ok with the performance gains of w8a16 as my model doesnt degrade much at this quantization level. Is there a way to acheive the same in SGLang?

Thanks



### Related resources

_No response_

---

## Issue #N/A: [Bug] compressed-tensors format not supported

**Link**: https://github.com/sgl-project/sglang/issues/2871
**State**: closed
**Created**: 2025-01-13T18:32:00+00:00
**Closed**: 2025-04-30T00:18:48+00:00
**Comments**: 5
**Labels**: inactive, quant

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

since [AutoFP8](https://github.com/neuralmagic/AutoFP8) has been deprecated in preference of [llm-compressor](https://github.com/vllm-project/llm-compressor), most recent quantization models are quantized by [llm-compressor](https://github.com/vllm-project/llm-compressor), but sglang does not support `compressed-tensors` format directly. f

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] DeepSeek V3 optimization

**Link**: https://github.com/sgl-project/sglang/issues/2591
**State**: closed
**Created**: 2024-12-26T08:52:39+00:00
**Closed**: 2025-03-25T04:10:46+00:00
**Comments**: 52
**Labels**: enhancement, high priority, performance, quant

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Adoption

[SGLang adoption for DeepSeek V3 and R1](https://github.com/sgl-project/sglang/discussions/3322)

### Usage

User Guide for Existing System (Installation & Launch)

https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3

Please use the latest version [v0.4.2.post4](https://pypi.org/project/sglang/0.4.2.post4/). Please prefer to use docker image. `docker pull lmsysorg/sglang:latest`

For running on AMD MI300X, use this as a reference. [Running DeepSeek-R1 on a single NDv5 MI300X VM](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/running-deepseek-r1-on-a-single-ndv5-mi300x-vm/4372726)

### Features

- [x] Support CUDA Graph @HandH1998 @ispobock 
- [x] Support Torch compile @is

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Add Docs For Quantization

**Link**: https://github.com/sgl-project/sglang/issues/2531
**State**: closed
**Created**: 2024-12-20T06:52:31+00:00
**Closed**: 2025-05-24T21:21:43+00:00
**Comments**: 5
**Labels**: good first issue, quant

### Description

Quick question, what is the recommended way to do offline quantization? I cannot find any documents on this. Thanks in advance!

---

## Issue #N/A: [Feature] Integrate CUTLASS FP8 GEMM into sgl-kernel

**Link**: https://github.com/sgl-project/sglang/issues/2472
**State**: closed
**Created**: 2024-12-12T20:08:31+00:00
**Closed**: 2025-02-12T00:16:40+00:00
**Comments**: 4
**Labels**: high priority, inactive, performance, quant

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

ref 
https://github.com/NVIDIA/cutlass/pull/1932/files

### Related resources

_No response_

---

