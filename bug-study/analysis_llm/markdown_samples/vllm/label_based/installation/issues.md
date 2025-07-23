# installation - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 26

### Label Distribution

- installation: 30 issues
- stale: 12 issues
- unstale: 2 issues
- rocm: 1 issues

---

## Issue #N/A: [Installation]: import llm meet error

**Link**: https://github.com/vllm-project/vllm/issues/4163
**State**: closed
**Created**: 2024-04-18T07:06:57+00:00
**Closed**: 2025-01-14T13:57:43+00:00
**Comments**: 6
**Labels**: installation, unstale

### Description

### Your current environment

```text
Traceback (most recent call last):
  File "inference.py", line 355, in <module>
    data_all_with_response = get_pred_func(data=data_all, task_prompt=task_prompt,\
  File "inference.py", line 24, in get_pred_vllm
    from vllm import LLM, SamplingParams
  File "/usr/local/lib/python3.8/dist-packages/vllm/__init__.py", line 3, in <module>
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
  File "/usr/local/lib/python3.8/dist-packages/vllm/engine/arg_utils.py", line 6, in <module>
    from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
  File "/usr/local/lib/python3.8/dist-packages/vllm/config.py", line 9, in <module>
    from vllm.utils import get_cpu_memory, is_hip
  File "/usr/local/lib/python3.8/dist-packages/vllm/utils.py", line 8, in <module>
    from vllm._C import cuda_utils
ImportError: /usr/local/lib/python3.8/dist-packages/vllm/_C.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops15to

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: I was never able to install it, which cuda version is required?

**Link**: https://github.com/vllm-project/vllm/issues/9960
**State**: closed
**Created**: 2024-11-02T22:53:06+00:00
**Closed**: 2025-03-03T02:03:22+00:00
**Comments**: 2
**Labels**: installation, stale

### Description

### Your current environment

I use ubunt 22.04

Installing this is almost impossible, what are actually requirements lets say for cuda. I spend many hours trying to install and never worked, there way always an error, something related to cuda version.


### How you are installing vllm

```sh
pip install -vvv vllm
```


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Installation]: Missing v0.6.3.post1-cu118-cp310.whl. Can share it? Thanks so much

**Link**: https://github.com/vllm-project/vllm/issues/10036
**State**: closed
**Created**: 2024-11-05T12:46:28+00:00
**Closed**: 2025-04-15T03:15:12+00:00
**Comments**: 4
**Labels**: installation, unstale

### Description

### Your current environment

Missing v0.6.3.post1-cu118-cp310.whl. Can share it? Thanks so much

### How you are installing vllm

Missing v0.6.3.post1-cu118-cp310.whl. Can share it? Thanks so much

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Installation]: Dockerfile.cpu installation problem vLLM

**Link**: https://github.com/vllm-project/vllm/issues/14033
**State**: closed
**Created**: 2025-02-28T10:02:14+00:00
**Closed**: 2025-07-02T02:14:03+00:00
**Comments**: 4
**Labels**: installation, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```
Dockerfile.cpu installation I can't complete the build somehow, I want to use vLLM over CPU since I don't have a graphics card on my own server, but the installation gives an error as follows.
OS= rockylinux 9.4 
ram 16gb
vCPU=> 24
Hypervisor= proxmox 8.2.7
docker version => Docker version 28.0.1, build 068a01e

errors messages;

docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .

Dockerfile.cpu:54
--------------------
  53 |     
  54 | >>> RUN --mount=type=cache,target=/root/.cache/pip \
  55 | >>>     --mount=type=cache,target=/root/.cache/ccache \
  56 | >>>     --mount=type=bind,source=.git,target=.git \
  57 | >>>     VLLM_TARGET_DEVICE=cpu python3 setup.py bdist_wheel && \
  58 | >>>     pip install dist/*.whl && \
  59 | >>>     rm -rf dist
  60 |     
--------------------
ERROR: failed to solve: process "/bin/sh -c VLLM_TARGET_DEVICE=cpu python3 setup.py bdist_wheel &&     pip install d

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: Installation instructions for ROCm can be mainlined

**Link**: https://github.com/vllm-project/vllm/issues/9385
**State**: closed
**Created**: 2024-10-15T18:21:50+00:00
**Closed**: 2025-02-13T01:59:19+00:00
**Comments**: 2
**Labels**: installation, stale

### Description

### Your current environment

N/A


### How you are installing vllm

https://docs.vllm.ai/en/stable/getting_started/amd-installation.html option 2

The problem is that it says to checkout a very specific commit of triton. Triton just published a new version of 3.1 that has AMD support mainlined but the dependency in the vllm pip package still tries to install 3.0. If someone tells me how I can update the dependency on 3.1 we can simplify the AMD instructions I think.

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Running to (Installing build dependencies ) this step is stuck

**Link**: https://github.com/vllm-project/vllm/issues/436
**State**: closed
**Created**: 2023-07-12T03:08:05+00:00
**Closed**: 2024-03-08T12:27:03+00:00
**Comments**: 5
**Labels**: installation

### Description

![image](https://github.com/vllm-project/vllm/assets/54533917/99997cfa-f6f6-4de1-9641-0e4c90884256)


system： ubuntu 20.04
Nvidia driver 515
cuda 11.7
rtx3090
python 3.8
vllm 0.1.2

---

## Issue #N/A: [Installation]: Unable to build docker image using Dockerfile.openvino

**Link**: https://github.com/vllm-project/vllm/issues/6769
**State**: closed
**Created**: 2024-07-25T04:05:59+00:00
**Closed**: 2024-07-30T18:33:02+00:00
**Comments**: 3
**Labels**: installation

### Description

### Your current environment

```text
(base) user@zahid:~/vllm$ python collect_env.py
Collecting environment information...
PyTorch version: N/A
Is debug build: N/A
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.2 LTS (x86_64)
GCC version: Could not collect
Clang version: Could not collect
CMake version: version 3.30.1
Libc version: glibc-2.35

Python version: 3.12.1 | packaged by Anaconda, Inc. | (main, Jan 19 2024, 15:51:05) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-1026-intel-iotg-x86_64-with-glibc2.35
Is CUDA available: N/A
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: Could not collect
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: N/A

CPU:
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Address sizes:   

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: Could not install packages due to an OSError: [Errno 28] No space left on device but disk still have space

**Link**: https://github.com/vllm-project/vllm/issues/7025
**State**: closed
**Created**: 2024-08-01T09:24:12+00:00
**Closed**: 2024-12-01T02:14:15+00:00
**Comments**: 3
**Labels**: installation, stale

### Description

### Your current environment

![image](https://github.com/user-attachments/assets/b25198d8-8530-49a1-b116-9882b5fb5977)
i install vllm in /mnt , i found is still have space but it has a wrong like:

Installing build dependencies ... error
  error: subprocess-exited-with-error
  
  × pip subprocess to install build dependencies did not run successfully.
  │ exit code: 1
  ╰─> [47 lines of output]
      Looking in indexes: http://mirrors.cloud.aliyuncs.com/pypi/simple/
      Collecting cmake>=3.21
        Using cached http://mirrors.cloud.aliyuncs.com/pypi/packages/78/5e/c274ffd124b8d4d95734af94c1080f0421c89dabdea2475651a7bd1e02ca/cmake-3.30.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (26.9 MB)
      Collecting ninja
        Using cached http://mirrors.cloud.aliyuncs.com/pypi/packages/6d/92/8d7aebd4430ab5ff65df2bfee6d5745f95c004284db2d8ca76dcbfd9de47/ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl (307 kB)
      Collecting packaging
   

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: no version of pip install vllm works - Failed to initialize NumPy: No Module named 'numpy'

**Link**: https://github.com/vllm-project/vllm/issues/11037
**State**: open
**Created**: 2024-12-09T22:11:26+00:00
**Comments**: 18
**Labels**: installation

### Description

### Your current environment

```text
Traceback (most recent call last):
  File "/mnt/MSAI/home/cephdon/sources/vllm/collect_env.py", line 15, in <module>
    from vllm.envs import environment_variables
  File "/mnt/MSAI/home/cephdon/sources/vllm/vllm/__init__.py", line 3, in <module>
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
  File "/mnt/MSAI/home/cephdon/sources/vllm/vllm/engine/arg_utils.py", line 11, in <module>
    from vllm.config import (CacheConfig, CompilationConfig, ConfigFormat,
  File "/mnt/MSAI/home/cephdon/sources/vllm/vllm/config.py", line 21, in <module>
    from vllm.model_executor.layers.quantization import (QUANTIZATION_METHODS,
  File "/mnt/MSAI/home/cephdon/sources/vllm/vllm/model_executor/__init__.py", line 1, in <module>
    from vllm.model_executor.parameter import (BasevLLMParameter,
  File "/mnt/MSAI/home/cephdon/sources/vllm/vllm/model_executor/parameter.py", line 7, in <module>
    from vllm.distributed import get_tensor_

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: Transformer installation requires uv venv --system now

**Link**: https://github.com/vllm-project/vllm/issues/15550
**State**: closed
**Created**: 2025-03-26T13:56:48+00:00
**Closed**: 2025-03-27T12:38:47+00:00
**Comments**: 2
**Labels**: installation

### Description

Hi all, a small one I can take care of is a breaking change introduced in 

https://github.com/vllm-project/vllm/commit/7ffcccfa5ca3ef6b56c292ad2489e077a5cdd6f5#diff-dd2c0eb6ea5cfc6c4bd4eac30934e2d5746747af48fef6da689e85b752f39557R62

The installation instructions in [here](https://docs.vllm.ai/en/latest/deployment/docker.html#deployment-docker-pre-built-image) should probably include `--system` as in:

`RUN uv pip install --system git+https://github.com/huggingface/transformers.git`

Thanks for your hard work!


### How you are installing vllm

```sh
pip install -vvv vllm
```


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Installation]:  issue with docker setup

**Link**: https://github.com/vllm-project/vllm/issues/9420
**State**: closed
**Created**: 2024-10-16T12:11:28+00:00
**Closed**: 2025-02-14T01:59:02+00:00
**Comments**: 3
**Labels**: installation, stale

### Description

### Your current environment

i am using unbuntu 22.04  having gpu of 16GB and ram of 34GB 
model is already download and reside at /home/ids/llm_models/zephyr
but after using below command I am getting error :
docker run --runtime nvidia --gpus all \
    -v /home/ids/llm_models/zephyr:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<key>" \
    -p 8080:8080 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model "/root/.cache/huggingface/zephyr-7b-alpha.Q8_0.gguf" \
    --load-format "gguf" \
    --dtype "float16" \
    --quantization "gguf" \
    --cpu-offload-gb 10 \
    --gpu-memory-utilization 0.5 \
    --max_seq_len_to_capture 4096 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --max-num-seqs 256 \
    --served-model-name "zephyr-7b-alpha"


"""
No module named 'vllm._version'
  from vllm.version import __version__ as VLLM_VERSION
INFO 10-16 05:09:17 api_server.py:528] vLLM API server version dev
INFO 10-16 05:09:17 api_server.py:529] a

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: Failed building editable for vllm

**Link**: https://github.com/vllm-project/vllm/issues/4913
**State**: closed
**Created**: 2024-05-20T07:14:29+00:00
**Closed**: 2024-09-19T12:18:37+00:00
**Comments**: 15
**Labels**: installation

### Description

### Your current environment

```
The output of `python collect_env.py`


Collecting environment information...

/usr/local/cuda-11.0/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn.so.8
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
/usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn.so.8
/usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
/usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
/usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
/usr/local/cuda-11.3/targets/x86_64-linux/lib/lib

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: How to install v0.9.2rc1 through Docker?

**Link**: https://github.com/vllm-project/vllm/issues/20483
**State**: open
**Created**: 2025-07-04T09:20:22+00:00
**Comments**: 1
**Labels**: installation

### Description

### Your current environment

How to install v0.9.2rc1 through Docker?


### How you are installing vllm




### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Building wheel for vllm (pyproject.toml) did not run successfully.

**Link**: https://github.com/vllm-project/vllm/issues/552
**State**: closed
**Created**: 2023-07-23T14:46:47+00:00
**Closed**: 2024-03-08T10:40:32+00:00
**Comments**: 4
**Labels**: installation

### Description

I'm trying to build a vLLM image on docker, but I keep getting an error while installing vllm module. I've tried with the given Dockerfile and I also try to build it from source but I keep getting this error. I'm not sure to understand what is the cause. 

Also, the installation of vllm module takes an hour, while it says it should take around 5/10min. It might come from my slow internet connexion.

I had to remove the beggining of the build log as it exceed max github length, but no error and no warning until building wheel.

<details>
<summary>Dockerfile</summary>

```
FROM nvcr.io/nvidia/pytorch:22.12-py3 AS base

EXPOSE 22/tcp
EXPOSE 8000/tcp

RUN python -m pip install --upgrade pip && \
    pip uninstall -y torch && \
    pip install vllm
```

</details>
<details>
  <summary>Docker build log</summary>
  
```
[+] Building 3580.7s (5/5) FINISHED
 => [internal] load build definition from Dockerfile                                                             

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: Install Gpu vllm got no module named triton

**Link**: https://github.com/vllm-project/vllm/issues/10244
**State**: closed
**Created**: 2024-11-12T06:26:47+00:00
**Closed**: 2025-03-20T02:03:49+00:00
**Comments**: 4
**Labels**: installation, stale

### Description

### Your current environment

```text
Install Gpu vllm 0.6.3 post6 with tesla v100and cuda 12.4,when import vllm got no module named triton
```


### How you are installing vllm

```sh
install with source code
```


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Installation]: Container image do not build Dockerfile.cpu

**Link**: https://github.com/vllm-project/vllm/issues/8502
**State**: closed
**Created**: 2024-09-16T13:58:50+00:00
**Closed**: 2024-09-18T02:49:54+00:00
**Comments**: 3
**Labels**: installation

### Description

### Your current environment

```text
$: podman -v
podman version 5.2.2
```


### How you are installing vllm

```sh
$: podman build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .
...
Error: building at STEP "RUN --mount=type=cache,target=/root/.cache/pip --mount=type=bind,src=requirements-build.txt,target=requirements-build.txt pip install --upgrade pip &&     pip install -r requirements-build.txt": resolving mountpoints for container "3a97f46183fa64e10c96f20f9a38a5ed46d2e9e7c4e7bbfbce6fa1adfdacd66e": invalid container path "requirements-build.txt", must be an absolute path
```

We can see that in the `Dockerfile.cpu` we are mounting the requirement-*.txt files

https://github.com/vllm-project/vllm/blob/fc990f97958636ce25e4471acfd5651b096b0311/Dockerfile.cpu#L29

https://github.com/vllm-project/vllm/blob/fc990f97958636ce25e4471acfd5651b096b0311/Dockerfile.cpu#L51

https://github.com/vllm-project/vllm/blob/fc990f97958636ce25e4471acfd5651b096b0311/Dockerfile.cpu

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: `Dockerfile.rocm` requires a torch nightly build that's no longer hosted: requires 2024-07-26, earliest is 2024-08-31.

**Link**: https://github.com/vllm-project/vllm/issues/9809
**State**: closed
**Created**: 2024-10-29T19:38:48+00:00
**Closed**: 2025-03-11T02:03:46+00:00
**Comments**: 3
**Labels**: installation, rocm, stale

### Description

### Your current environment

(current environment is irrelevant because this is a replacement for the nightly build reference)

### How you are installing vllm

```sh
git clone <vllm https URL>
cd vllm
git checkout -b 0.6.1 v0.6.1
podman build -f Dockerfile.rocm -t vllm-rocm .
```


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Error loading models since versions 0.6.1xxx

**Link**: https://github.com/vllm-project/vllm/issues/8745
**State**: closed
**Created**: 2024-09-23T21:17:29+00:00
**Closed**: 2024-10-18T11:11:35+00:00
**Comments**: 1
**Labels**: installation

### Description

### Your current environment

```
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: version 3.30.0
Libc version: glibc-2.31

Python version: 3.11.7 (main, Dec 15 2023, 18:12:31) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.4.0-187-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.3.107
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A40
GPU 1: NVIDIA A40
GPU 2: NVIDIA A40
GPU 3: NVIDIA A40

Nvidia driver version: 535.183.01
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Byte Order:                         Little Endian
Address

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: NotImplementedError get_device_capability

**Link**: https://github.com/vllm-project/vllm/issues/8243
**State**: closed
**Created**: 2024-09-06T17:42:09+00:00
**Closed**: 2024-09-10T14:02:21+00:00
**Comments**: 19
**Labels**: installation

### Description

### Your current environment

Collecting environment information...
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: AlmaLinux release 8.10 (Cerulean Leopard) (x86_64)
GCC version: (GCC) 8.5.0 20210514 (Red Hat 8.5.0-22)
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.28

Python version: 3.11.9 (main, Jul  2 2024, 16:32:17) [GCC 8.5.0 20210514 (Red Hat 8.5.0-22)] (64-bit runtime)
Python platform: Linux-4.18.0-553.8.1.el8_10.x86_64-x86_64-with-glibc2.28
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
cuDNN ver

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: https://hub.docker.com/u/vllm went wrong!

**Link**: https://github.com/vllm-project/vllm/issues/18328
**State**: closed
**Created**: 2025-05-19T01:43:13+00:00
**Closed**: 2025-05-19T06:20:13+00:00
**Comments**: 4
**Labels**: installation

### Description

### Your current environment

The vllm repo of docker hub went wrong.
<img width="1342" alt="Image" src="https://github.com/user-attachments/assets/841b974a-3825-4af1-843c-80ececfdced8" />

But other repos are good.
<img width="604" alt="Image" src="https://github.com/user-attachments/assets/946716ca-e8e3-4348-95ee-15d25e129cc9" />

### How you are installing vllm

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Installation]: poetry add vllm not working on my Mac -- xformers (0.0.26.post1) not supporting PEP 517 builds.

**Link**: https://github.com/vllm-project/vllm/issues/5690
**State**: closed
**Created**: 2024-06-19T17:41:35+00:00
**Closed**: 2025-03-30T02:10:11+00:00
**Comments**: 8
**Labels**: installation, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How you are installing vllm

```
poetry add vllm
```
Can someone tell me what versions of ray and torch are compatible to add vlllm via poetry?
```
- Installing vllm-flash-attn (2.5.9): Failed
-  Unable to find installation candidates for ray (2.24.0)
Note: This error originates from the build backend, and is likely not a problem with poetry but with xformers (0.0.26.post1) not supporting PEP 517 builds. You can verify this by running 'pip wheel --no-cache-dir --use-pep517 "xformers (==0.0.26.post1)"'
```


---

## Issue #N/A: [Installation]:

**Link**: https://github.com/vllm-project/vllm/issues/13427
**State**: closed
**Created**: 2025-02-17T21:54:28+00:00
**Closed**: 2025-02-18T05:54:37+00:00
**Comments**: 0
**Labels**: installation

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

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Installation]: Hard to find right wheel files to build the release version

**Link**: https://github.com/vllm-project/vllm/issues/18673
**State**: open
**Created**: 2025-05-25T03:17:17+00:00
**Comments**: 4
**Labels**: installation

### Description

@youkaichao @DarkLight1337 
Hello, I'm seeking help with building `v0.8.5.post1` and installing `other released versions`.

### Your current environment

trying to install v0.8.5.post1

### How you are installing vllm

https://github.com/vllm-project/vllm/issues/15347
Firstly, I had trouble identifying the correct commit IDs for the prebuilt wheel files from https://wheels.vllm.ai/
<img width="849" alt="Image" src="https://github.com/user-attachments/assets/f6a5966b-401d-4c3b-accf-0ed13e617558" />

https://github.com/vllm-project/vllm/issues/16217
Secondly, It would be very helpful if wheels were also published for the exact commits of the latest releases.
In my case, I’m trying to build v0.8.5.post1 using the prebuilt wheels. However, there is no wheel corresponding to the last commit ID of that release. (first try)
I then attempted to find the last commit of v0.8.5.post1 from the main branch and used the wheel for that commit, but encountered an import error. (second try)
I suspect t

[... truncated for brevity ...]

---

## Issue #N/A: [Installation] pip install vllm (0.6.3) will force a reinstallation of the CPU version torch and replace cuda torch on windows

**Link**: https://github.com/vllm-project/vllm/issues/9701
**State**: closed
**Created**: 2024-10-25T17:08:40+00:00
**Closed**: 2025-04-01T02:13:00+00:00
**Comments**: 50
**Labels**: installation, stale

### Description

pip install vllm (0.6.3) will force a reinstallation of the CPU version torch and replace cuda torch on windows. pip install vllm（0.6.3）将强制重新安装CPU版本的torch并在Windows上替换cuda torch。
> > 
> > 
> > I don't quite get what you mean, how can you have different versions of torch for CPU and GPU at the same time?我不太明白你的意思，你怎么能有不同版本的火炬CPU和GPU在同一时间？
> 
> only cuda torch
> 
> ```
>  pip install vllm --no-deps
> Collecting vllm
>   Using cached vllm-0.6.3.post1.tar.gz (2.7 MB)
>   Installing build dependencies ... error
>   error: subprocess-exited-with-error
> 
>   × pip subprocess to install build dependencies did not run successfully.
>   │ exit code: 2
>   ╰─> [86 lines of output]
>       Collecting cmake>=3.26
>         Using cached cmake-3.30.5-py3-none-win_amd64.whl.metadata (6.4 kB)
>       Collecting ninja
>         Using cached ninja-1.11.1.1-py2.py3-none-win_amd64.whl.metadata (5.4 kB)
> 
>       Collecting packaging
>         Using cached packaging-24.1-py3-none-any

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: Cannot install with Poetry

**Link**: https://github.com/vllm-project/vllm/issues/8851
**State**: closed
**Created**: 2024-09-26T13:02:53+00:00
**Closed**: 2025-02-07T01:59:47+00:00
**Comments**: 6
**Labels**: installation, stale

### Description

### Your current environment

5.47   /tmp/tmp4enbmtnv/.venv/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:258: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
45.47     cpu = _conversion_method_template(device=torch.device("cpu"))
45.47   Traceback (most recent call last):
45.47     File "/usr/local/lib/python3.12/site-packages/pyproject_hooks/_in_process/_in_process.py", line 373, in <module>
45.47       main()
45.47     File "/usr/local/lib/python3.12/site-packages/pyproject_hooks/_in_process/_in_process.py", line 357, in main
45.47       json_out["return_val"] = hook(**hook_input["kwargs"])
45.47                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
45.47     File "/usr/local/lib/python3.12/site-packages/pyproject_hooks/_in_process/_in_process.py", line 134, in get_requires_for_build_wheel
45.47       return hook(config_settings)
45.47              ^^^^^^^^

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: Hitting issues while trying to build vllm image using Dockerfile.rocm (v0.6.2)

**Link**: https://github.com/vllm-project/vllm/issues/11615
**State**: closed
**Created**: 2024-12-30T06:04:03+00:00
**Closed**: 2025-01-07T06:20:09+00:00
**Comments**: 1
**Labels**: installation

### Description

### Your current environment

Running docker build on ubuntu server.

Facing following error:
![image](https://github.com/user-attachments/assets/12e75246-2c48-4751-9484-d0182cceff50)

I checked https://download.pytorch.org/whl/nightly/torch/
But required wheel file: torch==2.6.0.dev20240918 and torchvision==0.20.0.dev20240918 is not present.


### How you are installing vllm

docker build -f Dockerfile.rocm -t vllm-rocm:0.6.2 .

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Installation]: offline chat gpt

**Link**: https://github.com/vllm-project/vllm/issues/10251
**State**: closed
**Created**: 2024-11-12T08:10:40+00:00
**Closed**: 2025-03-13T02:04:09+00:00
**Comments**: 3
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

## Issue #N/A: [Installation]: build docker images: Failed to build mamba-ssm

**Link**: https://github.com/vllm-project/vllm/issues/7498
**State**: closed
**Created**: 2024-08-14T02:48:22+00:00
**Closed**: 2024-10-11T15:56:30+00:00
**Comments**: 1
**Labels**: installation

### Description

### Your current environment

```
=> ERROR [mamba-builder 3/3] RUN pip --verbose wheel -r requirements-mamba.txt     --no-build-isolation --no-deps --no-cache-dir
909.8s
------
 > [mamba-builder 3/3] RUN pip --verbose wheel -r requirements-mamba.txt     --no-build-isolation --no-deps --no-cache-dir:
1.518 Collecting mamba-ssm>=1.2.2 (from -r requirements-mamba.txt (line 2))
1.776   Downloading mamba_ssm-2.2.2.tar.gz (85 kB)
1.904   Preparing metadata (setup.py): started
1.904   Running command python setup.py egg_info
4.070   No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
4.146
4.146
4.147   torch.__version__  = 2.4.0+cu121
4.147
4.148
4.149   running egg_info
4.149   creating /tmp/pip-pip-egg-info-d2_c2trw/mamba_ssm.egg-info
4.154   writing /tmp/pip-pip-egg-info-d2_c2trw/mamba_ssm.egg-info/PKG-INFO
4.154   writing dependency_links to /tmp/pip-pip-egg-info-d2_c2trw/mamba_ssm.egg-info/dependency_links.txt
4.155   writing requirements to /tmp/pip-pip-egg

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: RISC-V support?

**Link**: https://github.com/vllm-project/vllm/issues/8996
**State**: closed
**Created**: 2024-10-01T09:51:30+00:00
**Closed**: 2025-01-31T01:58:15+00:00
**Comments**: 3
**Labels**: installation, stale

### Description

### Your current environment

```text
PyTorch version: N/A
Is debug build: N/A
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: N/A

OS: Fedora Linux 38 (Thirty Eight) (riscv64)
GCC version: (GCC) 13.2.1 20230728 (Red Hat 13.2.1-1)
Clang version: Could not collect
CMake version: version 3.27.4
Libc version: glibc-2.37

Python version: 3.11.5 (main, Sep 15 2023, 00:00:00) [GCC 13.2.1 20230728 (Red Hat 13.2.1-1)] (64-bit runtime)
Python platform: Linux-6.1.80-riscv64-with-glibc2.37
Is CUDA available: N/A
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: Could not collect
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: N/A

CPU:
Architecture:        riscv64
Byte Order:          Little Endian
CPU(s):              128
On-line CPU(s) list: 0-127
NUMA node(s):        8
NUMA node0 CPU(s):   0-7,16-23
N

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]:  Cannot install vllm due to xformers:   ERROR: Failed building wheel for xformers  fatal: Not a git repository (or any parent up to mount point /scratch)  assert len(sources) > 0   AssertionError

**Link**: https://github.com/vllm-project/vllm/issues/17015
**State**: open
**Created**: 2025-04-23T00:09:22+00:00
**Comments**: 0
**Labels**: installation

### Description


Hi all, 

I am pip installing the latest vllm, 0.8.4.

CUDA: 12.4
torch: 2.6.0
Python: 3.12.1

```text
The output of `python collect_env.py`
```

python collect_env.py
Traceback (most recent call last):
  File "/program/ms-swift/collect_env.py", line 17, in <module>
    from vllm.envs import environment_variables
ModuleNotFoundError: No module named 'vllm'



I got several errors:

ERROR: Failed building wheel for xformers 
fatal: Not a git repository (or any parent up to mount point /scratch) 
assert len(sources) > 0 AssertionError 


Error logs:
----------------------------------------

(myvenv_msswift) [data@sh1 /program//ms-swift] $ pip install xformers
Collecting xformers
  Using cached xformers-0.0.29.post3.tar.gz (8.5 MB)


  Preparing metadata (setup.py) ... done
Building wheels for collected packages: xformers
  Building wheel for xformers (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
 

[... truncated for brevity ...]

---

