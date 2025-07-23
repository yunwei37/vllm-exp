# ray - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 12
- Closed Issues: 18

### Label Distribution

- ray: 30 issues
- bug: 21 issues
- stale: 11 issues
- tpu: 3 issues
- usage: 3 issues
- feature request: 3 issues
- documentation: 1 issues
- misc: 1 issues
- v1: 1 issues
- installation: 1 issues

---

## Issue #N/A: [Bug]: Vllm 0.8.2 + Ray 2.44 (Ray serve deployment) fallbacks to V0 Engine

**Link**: https://github.com/vllm-project/vllm/issues/15569
**State**: open
**Created**: 2025-03-26T19:29:21+00:00
**Comments**: 13
**Labels**: bug, ray

### Description

### Your current environment

<details>


```text
INFO 03-26 19:23:29 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.11.11 (main, Dec 11 2024, 16:28:39) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.8.0-1020-gcp-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.5.40
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A100-SXM4-40GB
Nvidia driver version: 550.90.07
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Addr

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM still runs after Ray workers crash

**Link**: https://github.com/vllm-project/vllm/issues/16259
**State**: open
**Created**: 2025-04-08T11:23:38+00:00
**Comments**: 12
**Labels**: bug, ray

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
INFO 04-08 04:09:19 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 4.0.0
Libc version: glibc-2.35

Python version: 3.12.9 (main, Feb  5 2025, 08:49:00) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-122-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H100 NVL
GPU 1: NVIDIA H100 NVL
GPU 2: NVIDIA H100 NVL
GPU 3: NVIDIA H100 NVL

Nvidia driver version: 555.52.04
cuDNN version: Could not collect
HIP runtime version: N/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Problems with vllm serve DeepSeek-R1 with 2 nodes and TP = 16（include vllm v0.8.4 v0.7.3 v0.7.2 V0 V1 engine）

**Link**: https://github.com/vllm-project/vllm/issues/16692
**State**: open
**Created**: 2025-04-16T02:43:26+00:00
**Comments**: 11
**Labels**: bug, ray

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>
 v0.8.4 using TP = 16 to serving deepseek-v3 in 2*H800*8 On Ray cluster, get EngineCore exception
```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

start command:
head node:
```bash
ray start --head --port=6379  && \
    vllm serve $MODELPATH \
    --max-num-seqs=256 \
    --max-model-len=32768 \
    --max-num-batched-tokens=32768 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --distributed-executor-backend=ray \
    --trust-remote-code \
    --served-model-name deepseek-r1
```
slave node:
```bash
ray start --block --address=$HEADPODIP:6379
```

get error:
```bash
2025-04-16 10:27:16,259 INFO usage_lib.py:467 -- Usage stats collection is enabled by default without user confirmation because this terminal is detected to be non-interactive. To disable this, add `--disa

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM on TPU does not support --pipeline-parallel-size with Ray

**Link**: https://github.com/vllm-project/vllm/issues/11260
**State**: open
**Created**: 2024-12-17T13:04:46+00:00
**Comments**: 4
**Labels**: bug, tpu, ray, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.6.0.dev20241126+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.2
Libc version: glibc-2.35

Python version: 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.5.0-1013-gcp-x86_64-with-glibc2.35
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
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:         

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Multi-Node CPU Inference on MacOS calling `intel_extension_for_pytorch` 

**Link**: https://github.com/vllm-project/vllm/issues/11342
**State**: closed
**Created**: 2024-12-19T17:56:00+00:00
**Closed**: 2025-02-05T07:11:03+00:00
**Comments**: 5
**Labels**: bug, ray

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### Model Input Dumps

_No response_

### 🐛 Describe the bug

While attempting to execute `NCCL_DEBUG=TRACE docker run -it --rm -p 8000:8000 --shm-size=4g --env "HUGGING_FACE_HUB_TOKEN=<token>" --env "VLLM_CPU_KVCACHE_SPACE=40" --privileged -e NCCL_IB_HCA=mlx5 vllm-cpu-env-latest --device="cpu" --disable_async_output_proc --enforce-eager --model=meta-llama/Llama-Guard-3-1B --dtype=float16 --tensor-parallel-size 16 --pipeline-parallel-size 2 --distributed-executor-backend="ray" --swap-space=1` for Multi-Node inference; It raise an error in `/usr/local/lib/python3.10/dist-packages/vllm/distributed/parallel_state.py` in `line 338` as it needs `intel_extension_for_pytorch` Module. 

The error disappeared after commenting lines 338 and 339, would be great for a quick fixing. 

Moreover, just want to confirm that R

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Worker died during distributed inference

**Link**: https://github.com/vllm-project/vllm/issues/15687
**State**: open
**Created**: 2025-03-28T07:41:20+00:00
**Comments**: 10
**Labels**: bug, ray, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

node1:

```text
INFO 03-28 00:32:32 [__init__.py:239] Automatically detected platform cuda.
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

Python version: 3.12.9 (main, Feb  5 2025, 08:49:00) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-130-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H20
GPU 1: NVIDIA H20
GPU 2: NVIDIA H20
GPU 3: NVIDIA H20
GPU 4: NVIDIA H20
GPU 5: NVIDIA H20
GPU 6: NVIDIA H20
GPU 7: NVIDIA H20

Nvidia driver version: 550.127.05
cuDNN version: Could not collect
HIP r

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Deepseek R1 failed to load with segfault when using multi-node serving in V1

**Link**: https://github.com/vllm-project/vllm/issues/17770
**State**: open
**Created**: 2025-05-07T07:50:44+00:00
**Comments**: 1
**Labels**: bug, ray

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

With vLLM image in docker hub `vllm/vllm-openai:v0.8.4`, attempt to run deepseek R1 model in V1 engine fail with segfault, while changing back to V0 using `export VLLM_USE_V1=0` can start successfully.

To reproduce, use 2 nodes and create a ray cluster according to vLLM's multi-node guide and run below command on one of the node
```
vllm serve deepseek-ai/DeepSeek-R1 --block-size 128 --max-model-len 3500 --max-num-batched-tokens 3500 --tensor-parallel-size 16 --disable-log-requests
```
Below segfault will be observed
```
(RayWorkerWrapper pid=2191173, ip=10.52.51.232) *** SIGSEGV received at time=1746566409 on cpu 131 ***
(RayWorkerWrapper pid=2191173, ip=10.52.51.232) PC: @     0x7fcfcd849b8a  (unknown)  addProxyOpIfNeeded()
(RayWorkerWrapper pid=2191173, ip=10.52.51.232)     @     0x

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Offline Inference Distributed

**Link**: https://github.com/vllm-project/vllm/issues/8966
**State**: closed
**Created**: 2024-09-30T13:29:52+00:00
**Closed**: 2025-04-20T02:11:02+00:00
**Comments**: 6
**Labels**: documentation, ray, stale

### Description

### 📚 The doc issue

Hi,

I was just wondering why in the "Offline Inference Distributed" example, `ds.map_batches()` is used. I used this initially, but I am now splitting the dataset and using `ray.remote()` which has the advantage that I don't need to specify the batch_size and can use continuous batching per GPU. 

### Suggest a potential alternative/fix

If useful I could contribute with an example in the docs with ray.remote(), so both methods are available

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: RAY_CGRAPH_get_timeout is not set successfully.  Ray still detects default timeout value.

**Link**: https://github.com/vllm-project/vllm/issues/19703
**State**: closed
**Created**: 2025-06-16T17:57:25+00:00
**Closed**: 2025-06-18T17:26:17+00:00
**Comments**: 2
**Labels**: bug, ray

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
INFO 06-16 10:48:23 [__init__.py:244] Automatically detected platform cuda.
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.5 LTS (aarch64)
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
P

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM-0.7.2 reports "No CUDA GPUs are available" while vllm-0.6.6.post1 works fine on kuberay under same environment conditions.

**Link**: https://github.com/vllm-project/vllm/issues/14415
**State**: closed
**Created**: 2025-03-07T07:12:23+00:00
**Closed**: 2025-07-08T02:14:11+00:00
**Comments**: 3
**Labels**: bug, ray, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 03-06 19:33:33 __init__.py:190] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.9.21 | packaged by conda-forge | (main, Dec  5 2024, 13:51:40)  [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.10.134-18.al8.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A10
Nvidia driver version: 550.144.03
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.1.0
/

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Ray connection timeout

**Link**: https://github.com/vllm-project/vllm/issues/19543
**State**: open
**Created**: 2025-06-12T09:54:23+00:00
**Comments**: 1
**Labels**: usage, ray

### Description

### Your current environment

docker run -d \
> --entrypoint /bin/bash \
> --network host \
        -v /data:/data \
        vllm:0.7.3-python3.11-cuda12.4-debian12 \
    -c "ray start --block --address=127.0.0.1:6379"> --runtime=nvidia \
> --name workernode \
> --shm-size 10.24g \
> --gpus "device=3" \
> -e VLLM_HOST_IP=127.0.0.1 \
> -e MASTER_ADDR=127.0.0.1 \
> -e MASTER_PORT=29500 \
> -v /data:/data \
> harbor.inspur.local/ai-group/vllm:0.7.3-python3.11-cuda12.4-debian12 \
>     -c "ray start --block --address=127.0.0.1:6379"
cb2dc97286d9e65858472bd6ce020f2694718b012dd6b2d89782fbf1e41eee05
(base) [root@nvidia-h20-1 ~]# docker logs -f workernode
[2025-06-13 01:49:41,730 E 1 1] gcs_rpc_client.h:179: Failed to connect to GCS at address 10.108.0.184:6379 within 5 seconds.

### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [x] Make sure you already searche

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Ray/vLLM RuntimeError: HIP error: invalid device ordinal

**Link**: https://github.com/vllm-project/vllm/issues/12572
**State**: closed
**Created**: 2025-01-30T08:35:34+00:00
**Closed**: 2025-06-11T02:14:01+00:00
**Comments**: 4
**Labels**: bug, ray, stale

### Description

### Your current environment



<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 01-30 09:13:33 __init__.py:187] No platform detected, vLLM is running on UnspecifiedPlatform
Collecting environment information...
PyTorch version: 2.5.0+rocm6.2
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.2.41133-dd7f95766

OS: Red Hat Enterprise Linux release 8.10 (Ootpa) (x86_64)
GCC version: (GCC) 8.5.0 20210514 (Red Hat 8.5.0-22)
Clang version: 17.0.6 (Red Hat 17.0.6-1.module+el8.10.0+20808+e12784c0)
CMake version: version 3.26.5
Libc version: glibc-2.28

Python version: 3.11.7 (main, Jun 17 2024, 15:34:21) [GCC 10.3.1 20210422 (Red Hat 10.3.1-1)] (64-bit runtime)
Python platform: Linux-4.18.0-553.22.1.el8_10.x86_64-x86_64-with-glibc2.28
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI250X (gfx90a:sramecc+:xnack-)
Nvidia driver version: C

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Speculative Decoding doesn't work with Ray compiled DAG and SPMD

**Link**: https://github.com/vllm-project/vllm/issues/13682
**State**: closed
**Created**: 2025-02-21T18:24:45+00:00
**Closed**: 2025-06-29T02:14:55+00:00
**Comments**: 7
**Labels**: feature request, ray, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

I'm trying to run DeepSeek-R1 using Ray's compiled DAG, SPMD, and enabled speculative decoding. I get an error after sending a request.

```
ERROR 02-21 02:30:26 async_llm_engine.py:68] Engine background task failed
ERROR 02-21 02:30:26 async_llm_engine.py:68] Traceback (most recent call last):
ERROR 02-21 02:30:26 async_llm_engine.py:68] File "/home/ray/anaconda3/lib/python3.12/site-packages/vllm/engine/async_llm_engine.py", line 58, in _log_task_completion
ERROR 02-21 02:30:26 async_llm_engine.py:68] return_value = task.result()
ERROR 02-21 02:30:26 async_llm_engine.py:68] ^^^^^^^^^^^^^
ERROR 02-21 02:30:26 async_llm_engine.py:68] File "/home/ray/anaconda3/lib/python3.12/site-packages/vllm/engine/async_llm_engine.py", line 825, in run_engine_loop
ERROR 02-21 02:30:26 async_llm_engine.py:68] resu

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: distributed using ray, how to get worker runtime error log

**Link**: https://github.com/vllm-project/vllm/issues/15514
**State**: closed
**Created**: 2025-03-26T01:58:53+00:00
**Closed**: 2025-03-31T15:15:28+00:00
**Comments**: 2
**Labels**: usage, ray

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: Running Tensor Parallel on TPUs on Ray Cluster

**Link**: https://github.com/vllm-project/vllm/issues/12058
**State**: closed
**Created**: 2025-01-14T21:32:12+00:00
**Closed**: 2025-01-24T05:41:50+00:00
**Comments**: 9
**Labels**: usage, tpu, ray

### Description

### Your current environment

```text
The output of `python collect_env.py`
The output of `python collect_env.py`
(test_hf_qwen pid=17527, ip=10.130.4.26) Environment Information:
(test_hf_qwen pid=17527, ip=10.130.4.26) Collecting environment information...
(test_hf_qwen pid=17527, ip=10.130.4.26) PyTorch version: 2.6.0.dev20241126+cpu
(test_hf_qwen pid=17527, ip=10.130.4.26) Is debug build: False
(test_hf_qwen pid=17527, ip=10.130.4.26) CUDA used to build PyTorch: None
(test_hf_qwen pid=17527, ip=10.130.4.26) ROCM used to build PyTorch: N/A
(test_hf_qwen pid=17527, ip=10.130.4.26) 
(test_hf_qwen pid=17527, ip=10.130.4.26) OS: Ubuntu 22.04.4 LTS (x86_64)
(test_hf_qwen pid=17527, ip=10.130.4.26) GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
(test_hf_qwen pid=17527, ip=10.130.4.26) Clang version: 14.0.0-1ubuntu1.1
(test_hf_qwen pid=17527, ip=10.130.4.26) CMake version: version 3.31.2
(test_hf_qwen pid=17527, ip=10.130.4.26) Libc version: glibc-2.35
(test_hf_qwen pid=17527, ip=10.13

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: ValueError: Ray does not allocate any GPUs on the driver node. Consider adjusting the Ray placement group or running the driver on a GPU node.

**Link**: https://github.com/vllm-project/vllm/issues/12983
**State**: closed
**Created**: 2025-02-09T10:53:54+00:00
**Closed**: 2025-06-26T02:26:05+00:00
**Comments**: 3
**Labels**: bug, ray, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

vllm容器：0.7.2
启动脚本：
bash run_cluster.sh \
    docker-hub.dahuatech.com/vllm/vllm-openai:v0.7.2 \
    10.12.167.20 \
    --head \
    /root/wangjianqiang/deepseek/DeepSeek-R1/DeepSeek-R1/ \
    -e VLLM_HOST_IP=$(hostname -I | awk '{print $1}')/ \
	-e "GLOO_SOCKET_IFNAME=ens121f0"/ \
    -e "NCCL_SOCKET_IFNAME=ens121f0"/ \
	-v /root/wangjianqiang/deepseek/DeepSeek-R1/:/root/deepseek_r1/

bash run_cluster.sh \
    docker-hub.dahuatech.com/vllm/vllm-openai:v0.7.2 \
    10.12.167.20 \
    --worker \
    /root/wangjianqiang/deepseek/DeepSeek-R1/DeepSeek-R1/ \
    -e VLLM_HOST_IP=$(hostname -I | awk '{print $1}')/ \
	-e "GLOO_SOCKET_IFNAME=ens121f0"/ \
    -e "NCCL_SOCKET_IFNAME=ens121f0"/ \
	-v /root/deepseek_r1/:/root/deepseek_r1/

启动命令：

root@admin:~/deepseek_r1/DeepSeek-R1# VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
root@admin:~/deep

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: running multiple vLLM instances on a single ray cluster

**Link**: https://github.com/vllm-project/vllm/issues/14277
**State**: closed
**Created**: 2025-03-05T10:43:43+00:00
**Closed**: 2025-07-04T02:17:47+00:00
**Comments**: 4
**Labels**: misc, ray, stale

### Description

### Anything you want to discuss about vllm.

When running the vLLM OpenAI server on a Ray cluster(with nodes A/B/C/D), I want to specify particular nodes(e.g., node A and B) for deployment, enabling better control over multiple vLLM instances within a single Ray cluster. Currently, it seems that Ray integration appears limited to specifying tp and pp.

Is supporting custom placement group a feasible option?

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [V1][Bug]: TP with Ray does not terminate gracefully

**Link**: https://github.com/vllm-project/vllm/issues/13437
**State**: closed
**Created**: 2025-02-17T23:37:17+00:00
**Closed**: 2025-02-19T17:40:51+00:00
**Comments**: 3
**Labels**: bug, ray, v1

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

When using Ray as the distributed executor backend and using the `LLM` Python API , the main process does not terminate gracefully:

```
*** SIGTERM received at time=1739834838 on cpu 88 ***
PC: @     0x7fe108d1f117  (unknown)  (unknown)
    @     0x7fe108cd0520  (unknown)  (unknown)
[2025-02-17 15:27:18,341 E 2669821 2669821] logging.cc:460: *** SIGTERM received at time=1739834838 on cpu 88 ***
[2025-02-17 15:27:18,341 E 2669821 2669821] logging.cc:460: PC: @     0x7fe108d1f117  (unknown)  (unknown)
[2025-02-17 15:27:18,341 E 2669821 2669821] logging.cc:460:     @     0x7fe108cd0520  (unknown)  (unknown)
2025-02-17 15:27:18,342 INFO compiled_dag_node.py:1867 -- Tearing down compiled DAG
2025-02-17 15:27:18,342 INFO compiled_dag_node.py:1872 -- Cancelling compiled worker on actor: Actor(RayWorkerW

[... truncated for brevity ...]

---

## Issue #N/A: Running Vllm on ray cluster, logging stuck at loading

**Link**: https://github.com/vllm-project/vllm/issues/5052
**State**: closed
**Created**: 2024-05-25T17:48:19+00:00
**Closed**: 2024-06-13T09:01:53+00:00
**Comments**: 6
**Labels**: bug, ray

### Description

### Your current environment

I have two machine 2*4090, I wanted to runner a model (eg gpt-neox-20b) using vllm on ray cluster, so i follow the documentation by making ray cluster 
on head
ray start --head
on node
ray start --address=<node-ip>:port

I manged to make the cluster so far, when i run simple script for inference: 
```
from vllm import LLM
llm = LLM("/home/administrator/nlp-deploy/models/gpt-neox-20b/", tensor_parallel_size=2, disable_custom_all_reduce=True, enforce_eager=True)

prompt = "GPT-NeoX-20B is"
output = llm.generate(prompt)
print(output)
```
the model is stuck at loading.
nvidia-smi for the head and the node while loading 
![image](https://github.com/vllm-project/vllm/assets/83926003/9d4adad6-e165-4e52-a2a5-3e281722586a)


logs when running the script 
![image](https://github.com/vllm-project/vllm/assets/83926003/57a6d968-fbf7-4f88-be90-895d73141058)

versions
cuda : 12.4
ray : 2.22

### 🐛 Describe the bug

i tried other solutions mention

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Can support CPU inference with Ray cluster?

**Link**: https://github.com/vllm-project/vllm/issues/15266
**State**: open
**Created**: 2025-03-21T03:13:09+00:00
**Comments**: 4
**Labels**: feature request, ray, stale

### Description

### 🚀 The feature, motivation and pitch

Can vllm support CPU inference with Ray cluster now? How to use it?



I have Ray cluster with two nodes, as follow:

```
======== Autoscaler status: 2025-03-21 02:55:20.232713 ========
Node status
---------------------------------------------------------------
Active:
 1 node_211c6676b30ed72f47827575ebf7360457841df4a44d8a09228163be
 1 node_d5b1b2aabbbed13ed9f8cd9111b6818c97e269d1c01c1378fab26e74
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 0.0/256.0 CPU
 0B/323.77GiB memory
 0B/142.75GiB object_store_memory

Demands:
 (no resource demands)

```


And I run vllm at only one node of the ray cluster, by:
```
VLLM_CPU_KVCACHE_SPACE=64 python3 -m vllm.entrypoints.openai.api_server --port 8080 --trust-remote-code --served-model-name QwQ32B --dtype float16 --model /root/LLM/QwQmodelscop/ --tensor-parallel-size 2
```


When I submit a inference task, ther

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: VLLM does not support TPU v5p-16 (Multi-Host) with Ray Cluster

**Link**: https://github.com/vllm-project/vllm/issues/10155
**State**: closed
**Created**: 2024-11-08T11:36:17+00:00
**Closed**: 2025-02-07T02:16:05+00:00
**Comments**: 12
**Labels**: installation, tpu, ray

### Description

### Your current environment

```text
The output of `python collect_env.py`

Collecting environment information...
WARNING:root:libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.
INFO 11-04 16:11:44 importing.py:15] Triton not installed or not compatible; certain GPU-related functions will not be available.
PyTorch version: 2.6.0
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 3.30.5
Libc version: glibc-2.31

Python version: 3.10.15 (main, Oct 17 2024, 02:58:23) [GCC 10.2.1 20210110] (64-bit runtime)
Python platform: Linux-5.19.0-1022-gcp-x86_64-with-glibc2.31
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen run

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Llama-4-Maverick crashes on V1 engine (with Ray distributed executor)

**Link**: https://github.com/vllm-project/vllm/issues/18023
**State**: open
**Created**: 2025-05-12T20:37:07+00:00
**Comments**: 9
**Labels**: bug, ray

### Description

### Your current environment

Forward passes with Maverick served across two 8xH100 nodes are crashing on V1 engine, but running fine on V0. 

Here are the steps to reproduce the error:
1. follow vllm docs from https://docs.vllm.ai/en/latest/serving/distributed_serving.html#running-vllm-on-multiple-nodes to enable vllm on multiple nodes (I'm using vllm/vllm-openai image with 0.8.5.post1)

2. serve the model with `--distributed-executor-backend ray` (without it, vllm engine falls back to V0):
```bash
vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct -tp 8 -pp 2 --dtype auto --max-model-len 4096 --gpu-memory-utilization 0.8 --enable-chunked-prefill --distributed-executor-backend ray
```

4. send some GSM8k requests (requires installing `pip install lmeval[api]`)
```bash
lm_eval \
  --model local-completions \
  --model_args model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",base_url=""http://localhost:8000/v1/completions"",max_retries=3,timeout=300,tokenized_requests=True,add_b

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][Ray]: Pipeline parallelism fails on the same host

**Link**: https://github.com/vllm-project/vllm/issues/14093
**State**: closed
**Created**: 2025-03-02T11:54:32+00:00
**Closed**: 2025-07-17T02:16:38+00:00
**Comments**: 8
**Labels**: bug, ray, stale

### Description

### Your current environment

Using the 0.7.3 ghcr.io/sasha0552/vllm:v0.7.3 (pascal docker from [pascal-pkgs-ci](https://github.com/sasha0552/pascal-pkgs-ci)) and the same version directly from vllm

Head:
```
docker run -d --rm \
	--entrypoint /bin/bash \
	--network host \
	--runtime=nvidia \
	--name headnode \
	--shm-size 10.24g \
	--gpus "device=0" \
	-e VLLM_HOST_IP=0.0.0.0 \
	-e MASTER_ADDR=127.0.0.1 \
	-e MASTER_PORT=29500 \
	-e NCCL_DEBUG=INFO \
	-e NCCL_SOCKET_IFNAME=br0 \
	-e NCCL_PCI_BUS_ID="00000000:01:00.0" \
	-v /media/bkutasi/60824A4F824A29BC/Other_projects/koboldcpp_precompiled:/models \
	vllm/vllm-openai:v0.7.3     -c "ray start --block --head --port=6379 --node-ip-address=0.0.0.0 --dashboard-host=0.0.0.0"
``` 
Worker:
```
docker run -d --rm \
	--entrypoint /bin/bash \
	--network host \
	--runtime=nvidia \
	--name workernode \
	--shm-size 10.24g \
	--gpus "device=1" \
	-e VLLM_HOST_IP=127.0.0.1 \
	-e MASTER_ADDR=127.0.0.1 \
	-e MASTER_PORT=29500 \
	-e NCCL_DEBUG=INFO \


[... truncated for brevity ...]

---

## Issue #N/A: [Bug]:  IndexError when using Ray-SPMD Worker with Multi-Step (--num-scheduler-steps > 1)

**Link**: https://github.com/vllm-project/vllm/issues/17904
**State**: open
**Created**: 2025-05-09T13:54:27+00:00
**Comments**: 1
**Labels**: bug, ray

### Description

### Your current environment

～

### 🐛 Describe the bug


###  Description
Running vLLM in **Ray-SPMD Worker** mode (`VLLM_USE_RAY_SPMD_WORKER=1`) together with **Multi-Step scheduling** (`--num-scheduler-steps 8`) crashes with an `IndexError` inside `MultiStepModelRunner.execute_model`.  
The error occurs while advancing to the next step: `model_input.cached_outputs[-1]` is empty, so accessing `[-1]` raises.

###  Reproduction

```bash
# single-GPU example; same happens with more GPUs
export VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_COMPILED_DAG =1

python -m vllm.entrypoints.openai.api_server \
  --model Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --distributed-executor-backend \
  ray \
  --num-scheduler-steps 8
```

Full stack trace (excerpt):

```
self.worker._execute_model_spmd(execute_model_req,
  File ".../vllm/worker/worker.py", line 399, in _execute_model_spmd
    output = super()._execute_model_spmd(execute_model_req,
  File ".../vllm/worker/worker_base.py", li

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Data parallel inference in offline mode(based on Ray)

**Link**: https://github.com/vllm-project/vllm/issues/14683
**State**: open
**Created**: 2025-03-12T14:14:42+00:00
**Comments**: 6
**Labels**: feature request, ray

### Description

### 🚀 The feature, motivation and pitch

I've been building model evaluation datasets using offline inference as outlined in the [documentation](https://docs.vllm.ai/en/stable/serving/offline_inference.html#offline-inference), and I noticed that it’s challenging to fully leverage all available GPUs—when the model fits on a single GPU.

To overcome this, I implemented a feature that distributes model replicas across different GPUs, allowing prompt data to be processed concurrently. For large datasets, this approach achieves nearly linear speedup, significantly enhancing performance for both my team and me.

It’s important to note that offline inference also plays a crucial role in model training and evaluation. By enabling efficient and scalable processing of evaluation data, offline inference helps in thoroughly benchmarking models and fine-tuning them during the development cycle.

Interestingly, this feature has been discussed before (see [issue #1237](https://github.com/vllm-project

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: No Cuda GPUs are available when running vLLM on Ray (Qwen 2.5 VL AWQ)

**Link**: https://github.com/vllm-project/vllm/issues/14456
**State**: closed
**Created**: 2025-03-07T19:55:44+00:00
**Closed**: 2025-07-09T02:15:35+00:00
**Comments**: 5
**Labels**: bug, ray, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Chainguard (x86_64)
GCC version: (Wolfi 14.2.0-r4) 14.2.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.40

Python version: 3.10.15 (tags/v3.10.15-0-gffee63f-dirty:ffee63f, Sep 23 2024, 21:00:09) [GCC 14.2.0] (64-bit runtime)
Python platform: Linux-5.10.227-219.884.amzn2.x86_64-x86_64-with-glibc2.40
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA L40S
GPU 1: NVIDIA L40S
GPU 2: NVIDIA L40S
GPU 3: NVIDIA L40S

Nvidia driver version: 550.127.05
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
/bin/sh: lscpu: not found

Versions of 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: LLM initialization time increases significantly with larger tensor parallel size and Ray

**Link**: https://github.com/vllm-project/vllm/issues/10283
**State**: closed
**Created**: 2024-11-13T04:08:02+00:00
**Closed**: 2024-12-19T07:38:03+00:00
**Comments**: 5
**Labels**: bug, ray

### Description

### Your current environment
vllm 0.5.2
<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.3.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.5 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
Clang version: Could not collect
CMake version: version 3.24.1
Libc version: glibc-2.31

Python version: 3.8.10 (default, Mar 13 2023, 10:26:41)  [GCC 9.4.0] (64-bit runtime)
Python platform: Linux-5.10.134-008.7.kangaroo.al8.x86_64-x86_64-with-glibc2.29
Is CUDA available: True
CUDA runtime version: 12.1.66
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L20Z
GPU 1: NVIDIA L20Z
GPU 2: NVIDIA L20Z
GPU 3: NVIDIA L20Z
GPU 4: NVIDIA L20Z
GPU 5: NVIDIA L20Z
GPU 6: NVIDIA L20Z
GPU 7: NVIDIA L20Z

Nvidia driver version: 535.161.08
cuDNN version: Probably one of the following:
/usr/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: RuntimeError: No CUDA GPUs are available | when using TP>1 and using vllm v0.7

**Link**: https://github.com/vllm-project/vllm/issues/14413
**State**: closed
**Created**: 2025-03-07T06:59:47+00:00
**Closed**: 2025-04-01T04:02:28+00:00
**Comments**: 5
**Labels**: bug, ray

### Description

### Your current environment

### Environment
```
root@7dc9530a8c13:/workspace# python collect_env.py 
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.5
Libc version: glibc-2.35

Python version: 3.11.9 | packaged by conda-forge | (main, Apr 19 2024, 18:36:13) [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-5.10.223-212.873.amzn2.x86_64-x86_64-with-glibc2.35
Is CUDA available: False
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: Could not collect
Nvidia driver version: Could not collect
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.1.0
/usr/lib/x86_64-linux

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: ray + vllm async engine: Background loop is stopped

**Link**: https://github.com/vllm-project/vllm/issues/7904
**State**: closed
**Created**: 2024-08-27T09:23:44+00:00
**Closed**: 2025-04-03T15:47:03+00:00
**Comments**: 14
**Labels**: bug, ray

### Description

### 🐛 Describe the bug
this code is slighly modified from [async llm engine test](https://github.com/vllm-project/vllm/blob/4cf256ae7f8b0be8f06f6b85821e55d4f5bdaa13/tests/async_engine/test_async_llm_engine.py#L115)

```
def test_asyncio_run():
    wait_for_gpu_memory_to_clear(
        devices=list(range(torch.cuda.device_count())),
        threshold_bytes=2 * 2**30,
        timeout_s=60,
    )

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model="facebook/opt-125m"))

    async def run(prompt: str):
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=32,
        )

        async for output in engine.generate(prompt,
                                            sampling_params,
                                            request_id=prompt):
            final_output = output
        return final_output

    async def generate():
        return await asyncio.gather(
            run("test0"),
      

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

