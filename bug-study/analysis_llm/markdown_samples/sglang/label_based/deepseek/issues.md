# deepseek - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 2
- Closed Issues: 28

### Label Distribution

- deepseek: 30 issues
- inactive: 11 issues
- help wanted: 9 issues
- high priority: 7 issues
- collaboration: 3 issues
- performance: 3 issues
- flashinfer: 2 issues
- amd: 1 issues
- MLLM: 1 issues
- enhancement: 1 issues

---

## Issue #N/A: Instruction for Running DeepSeek with PD, EP, and MTP

**Link**: https://github.com/sgl-project/sglang/issues/7998
**State**: open
**Created**: 2025-07-13T18:21:14+00:00
**Comments**: 3
**Labels**: deepseek

### Description

# Using Main Branch

## Environment Preparation
Use SGLang and DeepEP on master is sufficient. Also remember to upgrade Mooncake. It will be better to create customized expert distribution data for MTP (follow the related instructions in #6017)


## xP + 2D, max_running_requests=32, draft_token_num=3

### Command for decode
```
SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=10000000 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 MC_TE_METRIC=true python3 -m sglang.launch_server --model-path /mnt/shared-fs/models/deepseek-ai/DeepSeek-V3-0324 --disaggregation-ib-device mlx5_1 --disaggregation-mode decode --dist-init-addr 10.0.7.67:5757 --tp-size 16 --dp-size 16 --enable-dp-attention --decode-log-interval 1 --enable-deepep-moe --page-size 64 --host 0.0.0.0 --trust-remote-code --moe-dense-tp-size 1 --enable-dp-lm-head --disable-radix-cache --watchdog-timeout 1000000 --deepep-mode low_latency --mem-fraction-static 0.8 --max-running-requests 32 --cont

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support NVRTC for DeepGEMM

**Link**: https://github.com/sgl-project/sglang/issues/5313
**State**: closed
**Created**: 2025-04-12T06:17:24+00:00
**Closed**: 2025-05-13T08:45:22+00:00
**Comments**: 0
**Labels**: high priority, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

### Related resources

_No response_

---

## Issue #N/A: [Bug] sglang crashed when using /health_generate

**Link**: https://github.com/sgl-project/sglang/issues/3695
**State**: closed
**Created**: 2025-02-19T10:45:57+00:00
**Closed**: 2025-05-26T00:19:58+00:00
**Comments**: 4
**Labels**: inactive, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When deploying DeepSeek-V3/DeepSeek-R1 on two 8xH20 nodes, configuring a health probe (using health_generate) causes the service to crash after approximately 40 minutes.

the crash log:

```2025-02-19 11:45:37 TP0] Prefill batch. #new-seq: 1, #new-token: 1, #cached-token: 0, cache hit rate: 0.00%, token usage: 0.00, #running-req: 0, #queue

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Flashinferv0.2.2.post1 shows Unsupported max_mma_kv: 0 error on L40 , when deploying Deepseek-V2-Lite-chat with --enable-flashinfer-mla

**Link**: https://github.com/sgl-project/sglang/issues/4196
**State**: closed
**Created**: 2025-03-08T05:53:08+00:00
**Closed**: 2025-03-08T18:02:25+00:00
**Comments**: 1
**Labels**: deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

    return self.forward_extend(
    o, _ = self.prefill_wrapper_ragged.forward_return_lse(
    return self.run_return_lse(q, k, v)
    self._cached_module.ragged_run(*run_args)
    ragged_run_func(
    return self._op(*args, **(kwargs or {}))
    return func(*args, **kwargs)
    return self._op(*args, **(kwargs or {}))
RuntimeError: Error 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]  Two Node H20 with ROCE, can't startup

**Link**: https://github.com/sgl-project/sglang/issues/3603
**State**: closed
**Created**: 2025-02-16T06:31:52+00:00
**Closed**: 2025-02-18T05:34:59+00:00
**Comments**: 4
**Labels**: help wanted, high priority, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

sglang cant's start when using two node h20 to deploy.... hang!

### Reproduction

I m trying  to enhance my h20*2 inference efficiency by using ROCE. 
There were not prolblems using Socket Mode for NCCL.

Now , i'm using the 0.4.2 offiicial image and i have add some rdma pacakges in it according the commit: https://github.com/FrankLeeeee/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] CUDA error: uncorrectable ECC error encountered

**Link**: https://github.com/sgl-project/sglang/issues/3204
**State**: closed
**Created**: 2025-01-29T08:59:43+00:00
**Closed**: 2025-02-03T19:53:54+00:00
**Comments**: 3
**Labels**: deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
/usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.26.4
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
Max context length: 163840
2025-01-29 08:51:14.393720: E external/local_xla/xla/stream_executor/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] torch.distributed.all_reduce raised Segmentation fault on 2 * 8 * H800

**Link**: https://github.com/sgl-project/sglang/issues/3745
**State**: closed
**Created**: 2025-02-21T06:32:03+00:00
**Closed**: 2025-06-09T00:20:51+00:00
**Comments**: 5
**Labels**: inactive, deepseek

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

node 1 server Log: 


Fatal Python error: Segmentation fault

Thread 0x00007f2e93fff640 (most recent call first):
  File "/XXXX/sglang/python/sglang/srt/managers/scheduler.py", line 462 in watchdog_thread
  File "/usr/lib/python3.10/threading.py", line 953 in run
  File "/usr/lib/python3.10/threading.py", line 1016 in _bootstrap_inner
  Fi

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] run DeepSeek-R1 with --tp 2 --dp 2 --enable-dp-attention error

**Link**: https://github.com/sgl-project/sglang/issues/3667
**State**: closed
**Created**: 2025-02-18T09:56:09+00:00
**Closed**: 2025-06-13T00:19:53+00:00
**Comments**: 8
**Labels**: inactive, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

run DeepSeek-R1 with enable-dp-attention error

loc("/workspace/sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py":310:16): error: operation scheduled before its operands
loc("/workspace/sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py":310:16): error: operation scheduled before its operands
[

[... truncated for brevity ...]

---

## Issue #N/A: deepseek-v3 cannot run multi-node under H20

**Link**: https://github.com/sgl-project/sglang/issues/3398
**State**: closed
**Created**: 2025-02-08T09:07:16+00:00
**Closed**: 2025-05-27T00:18:50+00:00
**Comments**: 10
**Labels**: inactive, deepseek

### Description

hi there,

I have followed the doc in https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3, 
however the server was still stuck when setting up. 


**First node:**
ifconfig
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 11.130.1.53  netmask 255.255.248.0  broadcast 11.130.7.255

**Below is the command:**
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_DEBUG=TRACE

python3 -m sglang.launch_server --model-path /code/llm-benchmark-script/data/raw/DeepSeek-V3 --tp 16 --dist-init-addr 11.130.1.53:5000 --nnodes 2 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 6178

===========================

**Second node:**
ifconfig
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 33.18.27.52  netmask 255.255.248.0  broadcast 33.18.31.255

**Below is the command:**
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_DEBUG=TRACE

python3 -m sglang.launch_server --model-path /code/llm-be

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] After enabling flashinfer-mla for DeepSeek R1, I observed no throughput performance improvement.

**Link**: https://github.com/sgl-project/sglang/issues/4204
**State**: closed
**Created**: 2025-03-08T09:55:12+00:00
**Closed**: 2025-05-10T00:18:04+00:00
**Comments**: 4
**Labels**: inactive, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I found no performance improvement after enabling flashinfer-mla. In my environment (version 0.4.3-post2, H20*16, tp16), I compared 32 and 64 concurrency levels. The TTFT (Time to First Token) improved significantly:

‌64 concurrency‌: TTFT decreased from 19s to 14s, but throughput remained unchanged or slightly decreased.
‌32 concurrency‌

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Error when Load DeepSeek-R1 Model in --enable-ep-moe

**Link**: https://github.com/sgl-project/sglang/issues/3371
**State**: closed
**Created**: 2025-02-07T10:20:01+00:00
**Closed**: 2025-02-25T19:25:40+00:00
**Comments**: 5
**Labels**: deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Load DeepSeek-R1 Model error when --enable-ep-moe :
<img width="766" alt="Image" src="https://github.com/user-attachments/assets/2380b1e5-b55c-43bd-98a8-1ad0a740454a" />

### Reproduction

python3 -m sglang.launch_server --model-path /root/.cache/huggingface/models/DeepSeek-R1 --tp 16 --dist-init-addr ip:20000 --nnodes 2 --node-rank 0 --tr

[... truncated for brevity ...]

---

## Issue #N/A: Instruction for Running DeepSeek with Large-scale PD and EP

**Link**: https://github.com/sgl-project/sglang/issues/6017
**State**: open
**Created**: 2025-05-05T04:48:15+00:00
**Comments**: 504
**Labels**: collaboration, deepseek

### Description

## Using main branch

~~NOTE: The feature is already on main, but the performance still needs some improvements on main branch.~~ will be good after a few already opened PRs - PR 6680, 6727, 6728

~~NOTE: I will try other config like 4 node for P and 9 node for D later.~~ updated

### Environment Preparation

Use SGLang and DeepEP on master is sufficient. Also remember to upgrade Mooncake.

### 4P + 9D experiments

Start server
where DeepEP config can be tuned by https://github.com/sgl-project/sglang/pull/6742

```python
# prefill nodes
MC_TE_METRIC=true SGLANG_TBO_DEBUG=1 python3 -m sglang.launch_server --model-path /dev/shm/DeepSeek-V3-0324 --disaggregation-ib-device mlx5_1 --disaggregation-mode prefill --dist-init-addr 10.5.55.3:5757 --nnodes 4 --node-rank 0 --tp-size 32 --dp-size 32 --enable-dp-attention --decode-log-interval 1 --enable-deepep-moe --page-size 1 --host 0.0.0.0 --trust-remote-code --moe-dense-tp-size 1 --enable-dp-lm-head --disable-radix-cache --watchdog-timeout 1000

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] 4x8 Mi210 Deepseek V3 runtime error

**Link**: https://github.com/sgl-project/sglang/issues/3400
**State**: closed
**Created**: 2025-02-08T11:17:41+00:00
**Closed**: 2025-04-10T00:18:03+00:00
**Comments**: 2
**Labels**: inactive, amd, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to start 4 nodes with 32 Mi210 GPUs to launch Deepseek R1 using sglang and I've already converted the weights to bf16.

``` bash
[2025-02-08 18:30:47] INFO:     Started server process [445197]
[2025-02-08 18:30:47] INFO:     Waiting for application startup.
[2025-02-08 18:30:47] INFO:     Application startup complete.
[2025-02-08 1

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Optimizing DeepSeek with the DeepSeek Infra OSS component

**Link**: https://github.com/sgl-project/sglang/issues/3758
**State**: closed
**Created**: 2025-02-21T11:52:28+00:00
**Closed**: 2025-03-10T18:28:27+00:00
**Comments**: 4
**Labels**: high priority, performance, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

ref https://github.com/deepseek-ai/open-infra-index

- [ ] https://github.com/deepseek-ai/DeepEP
- [ ] https://github.com/deepseek-ai/DeepGEMM

### Related resources

_No response_

---

## Issue #N/A: [Bug] Deepseek-v3-0324 Error

**Link**: https://github.com/sgl-project/sglang/issues/5035
**State**: closed
**Created**: 2025-04-03T12:08:54+00:00
**Closed**: 2025-07-16T00:20:42+00:00
**Comments**: 2
**Labels**: inactive, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Failed to deploy Deepseek-v3-0324. 
(Deploy Deepseek-v3 successful.)

```
[2025-04-03 17:50:31 TP3] Scheduler hit an exception: Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/sglang/srt/managers/scheduler.py", line 1787, in run_scheduler_process
    scheduler = Scheduler(server_args, port_args, gpu_id, t

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] running requests low

**Link**: https://github.com/sgl-project/sglang/issues/4022
**State**: closed
**Created**: 2025-03-03T09:35:57+00:00
**Closed**: 2025-05-03T00:18:13+00:00
**Comments**: 1
**Labels**: help wanted, inactive, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When running the test script with 100 concurrent requests, the "running" count for sglang is relatively low, and there is a significant amount of data in the "queue", whereas the vllm service can reach a "running" count of 100.

### Reproduction

# evaluation script
```
evalscope perf \
    --url "http://192.168.8.**:31008/v1/chat/completi

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] attention backend default choice

**Link**: https://github.com/sgl-project/sglang/issues/5064
**State**: closed
**Created**: 2025-04-04T08:13:51+00:00
**Closed**: 2025-05-21T09:29:52+00:00
**Comments**: 2
**Labels**: high priority, collaboration, flashinfer, performance, MLLM, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

The standards we choose prioritize **performance first**, ease of use second (such as interface and installation), while also considering compatibility (such as older arch). Therefore, if in the future, the performance of different backends changes, we will still choose **the best performing one**.

1. NVIDIA

```
sm75 -> Triton
sm80, sm86, sm89 -> FlashInfer
sm90 -> FA3 (Llama, Qwen, Gemma), FlashInfer (Others)
sm100 -> FlashInfer

MLA
sm90 -> FA3 (DeepSeek)
sm100 -> FlashInfer (DeepSeek)

Other options
FlashMLA, cuDNN etc
```

SGLang will install the JIT version of FlashInfer on PyPI for a better user installation experience. Alternatively, the whl size limit of FlashInfer can be increased on PyPI. cc @yzh119 

F

[... truncated for brevity ...]

---

## Issue #N/A: [Tracker] SGLang v0.4.5.post1 performance on H200

**Link**: https://github.com/sgl-project/sglang/issues/5514
**State**: closed
**Created**: 2025-04-18T02:46:46+00:00
**Closed**: 2025-04-29T19:47:52+00:00
**Comments**: 9
**Labels**: high priority, collaboration, performance, deepseek

### Description

**Update**:
**see the latest benchmark results in another post https://github.com/sgl-project/sglang/pull/5611#issuecomment-2819965621** 


```bash
# launch server
# First, warm up for DeepGEMM
# SGLang uses FA3 backend by default since v0.4.5.post1
# Use dp 8 for offline use case
SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --enable-dp-attention --dp-size 8

# Random 1k, 2k
python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 50 --request-rate 10 --dataset-name random --random-input-len 1000 --random-output-len 2000 --random-range-ratio 1

# Random 5k, 1k
python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 50 --request-rate 10 --dataset-name random --random-input-len 5000 --random-output-len 1000 --random-range-ratio 1

# Random 10k, 500
python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 50 --request-rate 10 --dataset-name random --random-input-len 10000 --random-output

[... truncated for brevity ...]

---

## Issue #N/A: [Track] DeepSeek V3/R1 nextn progress

**Link**: https://github.com/sgl-project/sglang/issues/3472
**State**: closed
**Created**: 2025-02-10T14:46:03+00:00
**Closed**: 2025-03-25T04:13:25+00:00
**Comments**: 8
**Labels**: enhancement, high priority, flashinfer, deepseek

### Description

## Triton Backend

@ispobock @pankajroark 

- [x] [refactor triton backend 1](https://github.com/sgl-project/sglang/pull/3292), [2](https://github.com/sgl-project/sglang/pull/3309)

- [x] [support custom mask](https://github.com/sgl-project/sglang/pull/3317)

- [x] [support EAGLE 2](https://github.com/sgl-project/sglang/pull/3466)

- [x] [compatible with CUDA Graph](https://github.com/sgl-project/sglang/pull/3500)

- [x] [support nextn I (single MTP head)](https://github.com/sgl-project/sglang/pull/3582)

- [x] support next II (multi MTP heads) (WIP @pankajroark )

## FlashInfer Backend

@zhyncs @yzh119 

- [x] compatible with disable MLA

- [x] support FlashInfer nightly MLA ragged prefill and CUDA Core MLA decoding

- [x] support FlashInfer v0.2.0.post3 MLA ragged, paged prefill and decoding (@zhyncs @yzh119 )

- [x] nextn parts can be shared with Triton Backend

## EAGLE 2

@zhyncs @Ying1123 

- [x] implement sampling kernel in [sgl-kernel](https://github.com/sgl-project/sglang/tree

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

## Issue #N/A: i use 3 A800 to deploy deepseek r1，but one A800 just one  IB，how i adjust the number of tp in the deploy command

**Link**: https://github.com/sgl-project/sglang/issues/3517
**State**: closed
**Created**: 2025-02-12T08:40:00+00:00
**Closed**: 2025-02-13T19:23:45+00:00
**Comments**: 1
**Labels**: deepseek

### Description

# node 1
export NCCL_IB_HCA=mlx5_0
python3 -m sglang.launch_server --model-path /x32001214/model/bf16/DeepSeek-R1-BF16 --tp 12 --dist-init-addr 0.0.0.0:9997 --nnodes 3 --node-rank 0 --trust-remote-code  --host 0.0.0.0 --port 8888

# node 2
export NCCL_IB_HCA=mlx5_1
python3 -m sglang.launch_server --model-path /x32001214/model/bf16/DeepSeek-R1-BF16 --tp 24 --dist-init-addr 10.160.199.103:30172 --nnodes 3 --node-rank 1 --trust-remote-code

# node 3
export NCCL_IB_HCA=mlx5_1
python3 -m sglang.launch_server --model-path /x32001214/model/bf16/DeepSeek-R1-BF16 --tp 24 --dist-init-addr 10.160.199.103:30172 --nnodes 3 --node-rank 2 --trust-remote-code

---

## Issue #N/A: [Bug] Watchdog caught collective operation timeout:

**Link**: https://github.com/sgl-project/sglang/issues/3368
**State**: closed
**Created**: 2025-02-07T09:51:41+00:00
**Closed**: 2025-02-14T09:01:23+00:00
**Comments**: 8
**Labels**: deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Run DeepSeek-R1 on 2 8* A800s, 
This error occurs after loading the model:  Watchdog caught collective operation timeout:
10.25.117.26
```shell
[2025-02-07 02:59:26 TP13] Detected fp8 checkpoint. Please note that the format is experimental and subject to change.
[2025-02-07 02:59:26 TP11] Detected fp8 checkpoint. Please note that the forma

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]DeepSeek-R1 Process hangs after NCCL initialization in multi-server distributed inference setup

**Link**: https://github.com/sgl-project/sglang/issues/3516
**State**: closed
**Created**: 2025-02-12T08:37:39+00:00
**Closed**: 2025-02-19T12:42:30+00:00
**Comments**: 13
**Labels**: deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

### Environment and Setup
I am trying to run the DeepSeek-R1 671B model on three servers, each equipped with 8 A800 GPUs
I have 3 servers, each with 8 * A800 GPUs. I'm trying to create 5 nodes across these three servers using Docker overlay network for distributed inference, with each node using 4 GPUs. I've confirmed that all nodes can pi

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] DeepSeek R1 loading error by multi-node inference

**Link**: https://github.com/sgl-project/sglang/issues/3254
**State**: closed
**Created**: 2025-02-01T15:15:48+00:00
**Closed**: 2025-02-03T19:52:40+00:00
**Comments**: 10
**Labels**: help wanted, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to use multi-node inference (2 * 8 H100) to load DeepSeek R1.

However, it kept giving me error on the header node. **Note This error happens model weights are loaded 100%.**
```
Loading safetensors checkpoint shards: 100% Completed | 163/163 [1:09:28<00:00, 26.68s/it]
 
Loading safetensors checkpoint shards: 100% Completed | 163/1

[... truncated for brevity ...]

---

## Issue #N/A: NVIDIA L40*8  docker NCCL Hanging During Initialization on Single Node with Multiple GPUs

**Link**: https://github.com/sgl-project/sglang/issues/4054
**State**: closed
**Created**: 2025-03-04T07:13:01+00:00
**Closed**: 2025-05-04T00:21:09+00:00
**Comments**: 2
**Labels**: help wanted, inactive, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I am encountering an issue where NCCL hangs during initialization when running SGLang on a single node equipped with multiple GPUs (8 NVIDIA L40 GPUs). Despite my efforts to configure environment variables to prioritize shared memory (SHM) or NVLink over network communication, NCCL persists in using the NET/Socket mode and stalls at the In

[... truncated for brevity ...]

---

## Issue #N/A: [Usage] What's the best practice of deploying DeepSeekV3 using sglang?

**Link**: https://github.com/sgl-project/sglang/issues/4409
**State**: closed
**Created**: 2025-03-14T03:02:37+00:00
**Closed**: 2025-03-26T08:09:34+00:00
**Comments**: 1
**Labels**: deepseek

### Description

I want to run inference of [DeepSeekV3](https://huggingface.co/deepseek-ai/DeepSeek-V3) on multi-node GPU clusters. I have followed the [sglang deepseek guide](https://docs.sglang.ai/references/deepseek.html) to setup the distributed serving environment with the official sglang-v0.4.3.post4-cu124 docker image.

More specifically:
```
# node 1
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    --privileged \
    -v xxx:xxx \
    --name sglang_multinode1 \
    -it \
    --ipc=host \
    xxx

# node 2
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    --privileged \
    -v xxx:xxx \
    --name sglang_multinode2 \
    -it \
    --ipc=host \
    xxx
```
to create docker containers

```
# node 1
NCCL_IB_GID_INDEX=3 \
NCCL_DEBUG=TRACE \
NCCL_DEBUG=INFO \
python3 -m sglang.launch_server \
    --model-path xxx \
    --tp 16 --dist-init-addr xxx:20000 \
    --nnodes 2 \
    --node-rank 0 \
    --trust-remote-code \
    --host 0.0.0.0 --port 40000 2>&1 | tee 

[... truncated for brevity ...]

---

## Issue #N/A: [Error]Input length (160062 tokens) exceeds the maximum allowed length (59862 tokens).

**Link**: https://github.com/sgl-project/sglang/issues/4048
**State**: closed
**Created**: 2025-03-04T03:59:50+00:00
**Closed**: 2025-05-05T00:20:08+00:00
**Comments**: 12
**Labels**: help wanted, inactive, deepseek

### Description

Hi, I am trying to use `sglang` to deploy a DeepSeek R1 serving program. The deployment command is as follows:

```shell
python -m sglang.launch_server \
--model-path  /models/deepseek-ai/deepseek-r1 \
--host 0.0.0.0 \
--port 8100 \
--tensor-parallel-size 8 \
--mem-fraction-static 0.9 \
--trust-remote-code \
--context-length 163840 \
--chunked-prefill-size 4096 \
--served-model-name DeepSeek-R1-Sglang-160k
```

Although I set the `--context-length` parameter to 160k and the serving program starts successfully, an error occurs when I send a request with content of length 160k. The error message is as follows:

```shell
if self.sampling_params.max_new_tokens > 0:
TypeError: '>' not supported between instances of 'NoneType' and 'int'
[2025-03-03 06:49:28 TP5] Input length (160062 tokens) exceeds the maximum allowed length (59862 tokens). Use a shorter input or enable --allow-auto-truncate.
[2025-03-03 06:49:28 TP1] Input length (160062 tokens) exceeds the maximum allowed length (59862 tok

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] sglang crashes when serving DeepSeek-R1 with profiler enabled.

**Link**: https://github.com/sgl-project/sglang/issues/3815
**State**: closed
**Created**: 2025-02-24T11:09:31+00:00
**Closed**: 2025-02-26T10:52:39+00:00
**Comments**: 1
**Labels**: help wanted, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

One of the TP process crashes and the sglang server is down when I run `bench_serving.py` with `--profile`

possible direction: the two nodes have no NFS that can share the profile log folder. I am not sure if it's the problem. 


output of sglang
![Image](https://github.com/user-attachments/assets/7c16fde7-1308-4940-9547-f8260ab2fc2b)

ou

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support torch compile cache for DeepSeek V3/R1

**Link**: https://github.com/sgl-project/sglang/issues/3614
**State**: closed
**Created**: 2025-02-16T16:18:21+00:00
**Closed**: 2025-02-21T18:18:09+00:00
**Comments**: 6
**Labels**: good first issue, help wanted, high priority, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

The time taken for each startup is currently too long when torch compile is enabled. It needs optimization.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Improve Multi-node recipe to run inference

**Link**: https://github.com/sgl-project/sglang/issues/3206
**State**: closed
**Created**: 2025-01-29T12:21:53+00:00
**Closed**: 2025-01-31T23:48:24+00:00
**Comments**: 13
**Labels**: help wanted, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Could someone improve the example for serving DeepSeek 3 on multiple nodes adding information on how to run into a slurm cluster and singularity container.

https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208

### Related resources

_No response_

---

