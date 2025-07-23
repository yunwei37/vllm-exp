# Ascend_NPU - issues

**Total Issues**: 20
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 20

### Label Distribution

- Ascend NPU: 20 issues
- enhancement: 6 issues
- stale: 5 issues
- bug-unconfirmed: 3 issues
- low severity: 3 issues
- medium severity: 3 issues
- bug: 1 issues
- build: 1 issues
- critical severity: 1 issues

---

## Issue #N/A: Compile bug: Cann x86_64 not building

**Link**: https://github.com/ggml-org/llama.cpp/issues/12945
**State**: closed
**Created**: 2025-04-14T15:41:52+00:00
**Closed**: 2025-04-15T10:39:22+00:00
**Comments**: 3
**Labels**: bug, build, Ascend NPU

### Description

### Git commit

```
[ 13%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/llamafile/sgemm.cpp.o
/llama.cpp/ggml/src/ggml-cann/aclnn_ops.cpp: In function 'void ggml_cann_get_rows(ggml_backend_cann_context&, ggml_tensor*)':
/llama.cpp/ggml/src/ggml-cann/aclnn_ops.cpp:1786:49: error: 'float16_t' was not declared in this scope; did you mean 'float_t'?
 1786 |                 src0->data, ACL_FLOAT16, sizeof(float16_t), scale_ne, scale_nb,
      |                                                 ^~~~~~~~~
      |                                                 float_t
gmake[2]: *** [ggml/src/ggml-cann/CMakeFiles/ggml-cann.dir/build.make:90: ggml/src/ggml-cann/CMakeFiles/ggml-cann.dir/aclnn_ops.cpp.o] Error 1
gmake[2]: *** Waiting for unfinished jobs....
gmake[1]: *** [CMakeFiles/Makefile2:1790: ggml/src/ggml-cann/CMakeFiles/ggml-cann.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[ 13%] Linking CXX shared library ../../bin/libggml-cpu.so
[ 13%] Built target ggml

[... truncated for brevity ...]

---

## Issue #N/A: [CANN] Compile bug: no matching function for call to 'CastIntrinsicsImpl' Ascend NPU issues specific to Ascend NPUs

**Link**: https://github.com/ggml-org/llama.cpp/issues/12010
**State**: closed
**Created**: 2025-02-21T19:46:15+00:00
**Closed**: 2025-03-14T07:16:46+00:00
**Comments**: 4
**Labels**: Ascend NPU

### Description

I encountered the same issue(#10556 ) in Ascend310B1 as well. 
```
root@orangepiaipro-20t:/data/llama.cpp# cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release
-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- Including CPU backend
-- ARM detected
-- ARM feature FMA enabled
-- Adding CPU backend variant ggml-cpu:  
-- CANN: updated CANN_INSTALL_DIR from ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
-- CANN: SOC_VERSION auto-detected is:Ascend310B1
-- CANN: compile ascend kernels witch SOC_TYPE:Ascend310B1, SOC_VERSION:ascend310b1, compile macro:-DASCEND_310B.
-- CANN: CANN_INCLUDE_DIRS =  /usr/local/Ascend/ascend-toolkit/latest/include;/usr/local/Ascend/ascend-toolkit/latest/include/aclnn;/usr/local/Ascend/ascend-toolkit/latest/acllib/include
-- CANN: CANN_LIBRARIES =  ascendcl;nnopbase;opapi;acl_op_compiler;ascendc_kernels
-- Including CANN backend
-- Configu

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: CANN error  E89999 on Ascend 910b

**Link**: https://github.com/ggml-org/llama.cpp/issues/10777
**State**: closed
**Created**: 2024-12-11T07:06:39+00:00
**Closed**: 2025-01-21T08:49:47+00:00
**Comments**: 8
**Labels**: bug-unconfirmed, Ascend NPU

### Description

### Name and Version

./llama-cli  --version
version: 4302 (43041d2e)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for aarch64-linux-gnu

### Operating systems

Linux

### GGML backends

CANN

### Hardware

Huawei Ascend 910b

### Models

QwQ-32B-Q4_0

### Problem description & steps to reproduce

When I run the following command to start llama-cli, it crashed with CANN error CANN error: E89999: Inner Error!
 ```sh
 ./llama-cli -m /models/QwQ-32B-Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33 -sm layer
```



### First Bad Commit

_No response_

### Relevant log output

```shell
./llama-cli -m /models/QwQ-32B-Q4_0.ggup -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33 -sm layer
build: 4302 (43041d2e) with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for aarch64-linux-gnu
main: llama backend init
main: load the model and apply lora adapter, if any
llama_load_model_from_file: using d

[... truncated for brevity ...]

---

## Issue #N/A: [CANN] Compile bug: no matching function for call to 'CastIntrinsicsImpl'

**Link**: https://github.com/ggml-org/llama.cpp/issues/10556
**State**: closed
**Created**: 2024-11-28T03:48:05+00:00
**Closed**: 2024-11-29T06:46:03+00:00
**Comments**: 44
**Labels**: Ascend NPU

### Description

### Git commit

https://github.com/ggerganov/llama.cpp/commit/9f912511bc9414fa7a3c521378b6388cd932b58d

### Operating systems

Linux

### GGML backends

CPU

### Problem description & steps to reproduce

ggml CANN backend 

Ascend NPU: 910A or 910B.   not 910b1, 910b2, 910b3

Ascend-hdk-910-npu-driver_23.0.0_linux-aarch64.run --quiet --docker 
Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run --force --quiet --install-for-all --full 
Ascend-cann-kernels-910_8.0.RC2_linux.run --install --install-for-all --quiet 


cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release -DSOC_TYPE=ascend910a
or
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release -DSOC_TYPE=ascend910b


cmake --build build --config release --target llama-cli



```log
[ 41%] Performing build step for 'ascendc_kernels_device'
[ 12%] Building CXX object CMakeFiles/device_obj.dir/home/ma-user/llama.cpp/build/auto_gen/ascendc_kernels/auto_gen_dup.cpp.o
[ 25%] Building CXX object CMakeFiles/device_obj.di

[... truncated for brevity ...]

---

## Issue #N/A: [CANN] Compile bug:  cann backend build failed when manually specify SOC_TYPE or gcc version that isn't verified

**Link**: https://github.com/ggml-org/llama.cpp/issues/10517
**State**: closed
**Created**: 2024-11-26T13:36:14+00:00
**Closed**: 2024-11-28T07:26:20+00:00
**Comments**: 0
**Labels**: Ascend NPU

### Description

### Git commit

ab96610b1e58684bc5e8b810130c4cf6d8252e21

### Operating systems

Linux

### GGML backends

CANN

### Problem description & steps to reproduce

cann backend build failed when manually specify SOC_TYPE or gcc version that isn't verified

### First Bad Commit

c18610b4ee29ca056bb4f2d375a4ad1b16f44ef7

### Relevant log output

```shell
int4b_t is not supported.
```


---

## Issue #N/A: [CANN] Operator support

**Link**: https://github.com/ggml-org/llama.cpp/issues/10512
**State**: closed
**Created**: 2024-11-26T09:35:07+00:00
**Closed**: 2025-01-13T01:07:30+00:00
**Comments**: 1
**Labels**: enhancement, stale, Ascend NPU

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

> **This issue summarizes the current support of various operators in the CANN backend.**



### Precision issue

> This part is a newly added test case related to matrix transposition, which is pending fix.

```
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=256,bs=[2,3],nr=[1,1],per=[0,2,1,3]): [MUL_MAT] NMSE = 1.826328661 > 0.000500000 FAIL
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=256,bs=[2,3],nr=[1,1],per=[0,1,3,2]): [MUL_MAT] NMSE = 1.489608079 > 0.000500000 FAIL
MUL_MAT(type

[... truncated for brevity ...]

---

## Issue #N/A: Bug: 【CANN】ggml-cann/aclnn_ops.cpp:3007: GGML_ASSERT(n_dims == src0->ne[0]) failed

**Link**: https://github.com/ggml-org/llama.cpp/issues/10451
**State**: closed
**Created**: 2024-11-22T03:57:43+00:00
**Closed**: 2024-11-29T01:05:00+00:00
**Comments**: 8
**Labels**: Ascend NPU

### Description

### What happened?

按照readme中的固件驱动版本，推理时出现报错

### Name and Version

最新版本

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
llama_new_context_with_model: freq_scale    = 1
llama_new_context_with_model: n_ctx_per_seq (4096) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_kv_cache_init:      CANN0 KV buffer size =   132.00 MiB
llama_kv_cache_init:        CPU KV buffer size =    28.00 MiB
llama_new_context_with_model: KV self size  =  160.00 MiB, K (f16):   80.00 MiB, V (f16):   80.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.58 MiB
llama_new_context_with_model:      CANN0 compute buffer size =  1488.00 MiB
llama_new_context_with_model:  CANN_Host compute buffer size =    16.01 MiB
llama_new_context_with_model: graph nodes  = 1606
llama_new_context_with_model: graph splits = 67 (with bs=512), 3 (with bs=1)
common_init_from_params: warming up the model with 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: CANN  E89999

**Link**: https://github.com/ggml-org/llama.cpp/issues/10161
**State**: closed
**Created**: 2024-11-04T09:49:12+00:00
**Closed**: 2025-01-27T01:07:16+00:00
**Comments**: 22
**Labels**: bug-unconfirmed, stale, low severity, Ascend NPU

### Description

### What happened?

common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
/owner/ninth/llama.cpp/ggml/src/ggml-cann.cpp:61: CANN error: E89999: Inner Error!
E89999: [PID: 2277481] 2024-11-04-17:38:30.068.533 op[Range], outSize from framework (OFF) is 1, but outSize from tiling (OFT) is 64,which maybe calc OFF by double, but calc OFT by floatplease use float to calc OFF while you wanner input's dtype is float[FUNC:CalculateOutputNum][FILE:range.cc][LINE:113]
        TraceBack (most recent call last):
       op[Range], calculate output_total_num value fail.[FUNC:AppendTilingArgs][FILE:range.cc][LINE:182]
       op[Range], append tiling args fail.[FUNC:Tiling4Range][FILE:range.cc][LINE:255]
       Tiling failed
       Tiling Failed.
       Kernel Run failed. opType: 7, Range
       launch failed for Range, errno:561103.

  current device: 0, in function aclnn_arange at /owner/ninth/llama.cpp/ggml/src/ggml-cann/aclnn_ops.cpp:2

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: [CANN] backend supports Ascend 310P

**Link**: https://github.com/ggml-org/llama.cpp/issues/10160
**State**: closed
**Created**: 2024-11-04T09:35:09+00:00
**Closed**: 2024-11-22T06:07:41+00:00
**Comments**: 0
**Labels**: enhancement, Ascend NPU

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

CANN backend supports Ascend 310P inference accelerator card. Currently, llama.cpp already supports Ascend 910B. However, some APIs of Ascend 910B are different from those of 310P, so they need to be adapted in CANN backend implementation.

### Motivation

Compare to Ascend 910, Ascend 310 focuses on power-efficient inference on edge devices, The basic information as following:
**Inference-Oriented**: The 310P is optimized for **inference tasks**, focusing more on efficient and low-power operati

[... truncated for brevity ...]

---

## Issue #N/A: Bug: ascend 310p cann fatal error

**Link**: https://github.com/ggml-org/llama.cpp/issues/10108
**State**: closed
**Created**: 2024-10-31T12:16:06+00:00
**Closed**: 2025-01-13T01:07:34+00:00
**Comments**: 2
**Labels**: stale, Ascend NPU

### Description

### What happened?

ascend 310p cannot run model

### Name and Version

version: b2989
os:openeuler22.03 aarch64
backend:cann
chip:ascend 310p

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
app/ggml/src/ggml-cann.cpp:63: CANN error
CANN error: EZ9999: Inner Error!
EZ9999: 2024-10-31-12:08:42.276.420  The error from device(0), serial number is 3, there is an aivec error, core id is 0, error code = 0x10, dump info: pc start: 0x8001240801311a4, current: 0x124080131744, vec error info: 0x1ebe7caf, mte error info: 0, ifu error info: 0xfdf646ef6900, ccu error info: 0x6b8e3406005b7aca, cube error info: 0, biu error info: 0, aic error mask: 0x6de01200c0122c8, para base: 0x124000262840, errorStr: Illegal instruction, which is usually caused by unaligned UUB addresses.[FUNC:PrintCoreErrorInfo][FILE:device_error_proc.cc][LINE:537]
        TraceBack (most recent call last):
        The error from device(0), serial number is 3, there is an

[... truncated for brevity ...]

---

## Issue #N/A: Bug: b3990 ascend cann build error

**Link**: https://github.com/ggml-org/llama.cpp/issues/10105
**State**: closed
**Created**: 2024-10-31T02:04:51+00:00
**Closed**: 2024-11-04T11:08:40+00:00
**Comments**: 4
**Labels**: low severity, Ascend NPU

### Description

### What happened?

27.81 /app/ggml/src/ggml-cann.cpp: In function 'ggml_backend_buffer* ggml_backend_cann_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t, size_t)':
27.81 /app/ggml/src/ggml-cann.cpp:1230:19: error: 'struct ggml_backend_buffer_i' has no member named 'get_name'; did you mean 'get_base'?
27.81  1230 |     buffer->iface.get_name = ggml_backend_cann_host_buffer_name;
27.81       |                   ^~~~~~~~
27.81       |                   get_base
27.85 gmake[3]: *** [ggml/src/CMakeFiles/ggml.dir/build.make:174: ggml/src/CMakeFiles/ggml.dir/ggml-cann.cpp.o] Error 1
27.85 gmake[2]: *** [CMakeFiles/Makefile2:1619: ggml/src/CMakeFiles/ggml.dir/all] Error 2
27.85 gmake[1]: *** [CMakeFiles/Makefile2:3368: examples/main/CMakeFiles/llama-cli.dir/rule] Error 2
27.85 gmake: *** [Makefile:1323: llama-cli] Error 2
------


### Name and Version

version: b3990
os: openeuleros:22.03
framework: ascend-cann

### What operating system are you seeing the problem on?

M

[... truncated for brevity ...]

---

## Issue #N/A: Bug: [CANN] inference running result is garbled in debug running model for LM models who's type is Q4_0 class

**Link**: https://github.com/ggml-org/llama.cpp/issues/9979
**State**: closed
**Created**: 2024-10-21T11:35:28+00:00
**Closed**: 2024-10-22T08:16:03+00:00
**Comments**: 0
**Labels**: medium severity, Ascend NPU

### Description

### What happened?

For CANN backend: inference running result is garbled in debug running model for LM models who's type is Q4_0 class

### Name and Version

b3948

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Feature Request: [CANN] backend adapts to llama.cpp dynamic backend loading mechanism

**Link**: https://github.com/ggml-org/llama.cpp/issues/9862
**State**: closed
**Created**: 2024-10-12T09:25:05+00:00
**Closed**: 2024-10-22T08:16:32+00:00
**Comments**: 0
**Labels**: enhancement, Ascend NPU

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Dynamically loadable backends framework has been added in PR([#9707](https://github.com/ggerganov/llama.cpp/pull/9707)). CANN backend needs to adapt to this mechanism.

### Motivation

llama.cpp will be refactored to use only the backend registry API, as explained by slaren in PR ([#9707](https://github.com/ggerganov/llama.cpp/pull/9707)). Currently, CUDA and CPU backends has implemented these interfaces.

### Possible Implementation

CANN already implement the functions in these interfaces, so t

[... truncated for brevity ...]

---

## Issue #N/A: Bug: [CANN] compile failure 

**Link**: https://github.com/ggml-org/llama.cpp/issues/9844
**State**: closed
**Created**: 2024-10-11T10:27:37+00:00
**Closed**: 2024-10-16T00:52:51+00:00
**Comments**: 2
**Labels**: medium severity, Ascend NPU

### Description

### What happened?

# Version
lastest b3906

# System Info
Device: Ascend 910B4
OS: EulerOS 2.10  
Arch: aarch64

# What happened
follow the [CANN.md](https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/CANN.md) try to build llama-cli 
facing compile failure

logs
```
/app/ggml/src/ggml-common.h:261:16: warning: ISO C++ prohibits anonymous structs [-Wpedantic]
  261 |         struct {
      |                ^
/app/ggml/src/ggml-common.h:288:16: warning: ISO C++ prohibits anonymous structs [-Wpedantic]
  288 |         struct {
      |                ^
/app/ggml/src/ggml-common.h:305:16: warning: ISO C++ prohibits anonymous structs [-Wpedantic]
  305 |         struct {
      |                ^
/app/ggml/src/ggml-cann.cpp: In function 'ggml_backend_buffer_type* ggml_backend_cann_buffer_type(int32_t)':
/app/ggml/src/ggml-cann.cpp:1154:13: error: no match for 'operator=' (operand types are 'ggml_backend_buffer_type' and '<brace-enclosed initializer list>'

[... truncated for brevity ...]

---

## Issue #N/A: [CANN]Bug: Can't compile ggml/src/CMakeFiles/ggml.dir/ggml-cann/acl_tensor.cpp.o

**Link**: https://github.com/ggml-org/llama.cpp/issues/9560
**State**: closed
**Created**: 2024-09-20T07:08:32+00:00
**Closed**: 2024-11-22T06:07:58+00:00
**Comments**: 2
**Labels**: enhancement, Ascend NPU

### Description

### What happened?

After using 'cmake --build build --config release' command on Ascend 310P3,it can not compile succesfully
![image](https://github.com/user-attachments/assets/74ef8d67-e859-4502-ac0e-295513280fe3)


### Name and Version

# NPU
![image](https://github.com/user-attachments/assets/7e3c7184-7c72-462c-a5a5-fdfca2e57fbf)
# tookit
Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run
# kernels
Ascend-cann-kernels-310p_8.0.RC2_linux.run
# gcc
8.5.0
# platform
Euler OS 2.0
# llama.cpp
branch master
commitID 0d2f22e45c3c3b6f8222acb6284d0c8c93443ba1

### What operating system are you seeing the problem on?

Linux, Other? (Please let us know in description)

### Relevant log output

```shell
(PyTorch-2.1.0) [ma-user llama.cpp-master]$cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release
-- The CXX compiler identification is GNU 8.5.0
-- Check for working CXX compiler: /usr/bin/g++
-- Check for working CXX compiler: /usr/bin/g++ -- works
-- Detecting CXX compiler ABI

[... truncated for brevity ...]

---

## Issue #N/A: [CANN]Feature Request: Support OrangeAIPRO 310b CANN

**Link**: https://github.com/ggml-org/llama.cpp/issues/9481
**State**: closed
**Created**: 2024-09-14T09:33:14+00:00
**Closed**: 2024-11-02T01:07:14+00:00
**Comments**: 6
**Labels**: enhancement, stale, Ascend NPU

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Follow-up on issue https://github.com/ggerganov/llama.cpp/issues/9423, I hope that llama.cpp can support running on the Orange Pi AI PRO.

### Motivation

Currently, llama.cpp has incomplete support for CANN. There are quite a few Orange Pi AI PRO users, many of whom have a need for deploying large models.

### Possible Implementation

_No response_

---

## Issue #N/A: Feature Request: Add Host buffer type for Ascend NPU (CANN backend)

**Link**: https://github.com/ggml-org/llama.cpp/issues/9304
**State**: closed
**Created**: 2024-09-04T01:47:34+00:00
**Closed**: 2024-09-14T02:18:26+00:00
**Comments**: 2
**Labels**: enhancement, Ascend NPU

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Ascend NPU backend (CANN) is not support pin memory(Host buffer type) now. Using ping memory will make it more efficiency.

### Motivation

Other backend such as CUDA has already support Host buffer type.

### Possible Implementation

Refer to CUDA to implement the Host buffer type of Ascend NPU.

---

## Issue #N/A: Bug: Error when running a non-exist op for Ascend NPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/9303
**State**: closed
**Created**: 2024-09-04T01:32:43+00:00
**Closed**: 2024-09-12T01:02:36+00:00
**Comments**: 0
**Labels**: low severity, Ascend NPU

### Description

### What happened?

If execute a op that not exist, CANN backend will throw an error that NPU's context pointer is null.
The reason is that when op is not exist, context will not init, although it will only happed in test, but I think it's need to fix.

this command will reproduce this issue:
```
./test-backend-ops test -b CANN0 -o NOT_EXISTS
```

### Name and Version

version: 3662 (7605ae7d)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for aarch64-linux-gnu

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
~/code/llama.cpp/build/bin$ ./test-backend-ops test -b CANN0 -o NOT_EXISTS
Testing 3 backends

Backend 1/3 (CPU)
  Skipping
Backend 2/3 (CANN0)
  Backend name: CANN0
  1342/1342 tests passed
  Backend CANN0: OK

CANN error: EE1001: [PID: 205631] 2024-09-04-01:19:57.687.508 The argument is invalid.Reason: rtDeviceSynchronize execute failed, reason=[context pointer null]
        Solution: 1.Check the i

[... truncated for brevity ...]

---

## Issue #N/A: Bug: A crash occurs when llama-bench is running on multiple cann devices.

**Link**: https://github.com/ggml-org/llama.cpp/issues/9250
**State**: closed
**Created**: 2024-08-30T09:16:48+00:00
**Closed**: 2024-10-22T08:17:24+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale, critical severity, Ascend NPU

### Description

### What happened?

when i use  Llama3-8B-Chinese-Chat-f16-v2_1.gguf to run llama.cpp, here is a crash:
here is my cmd:
./llama-cli -m /home/c00662745/llama3/llama3/llama3_chinese_gguf/Llama3-8B-Chinese-Chat-f16-v2_1.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33 -sm layer

here is the error:
’‘’
CANN error: EE9999: Inner Error!
EE9999: [PID: 2750884] 2024-08-30-16:20:38.196.490 Stream destroy failed, stream is not in current ctx, stream_id=2.[FUNC:StreamDestroy][FILE:api_impl.cc][LINE:1032]
        TraceBack (most recent call last):
       rtStreamDestroy execute failed, reason=[stream not in current context][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
       destroy stream failed, runtime result = 107003[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]

  current device: 1, in function ~ggml_backend_cann_context at /home/zn/new-llama/llama.cpp/ggml/src/ggml-cann/common.h:235
  aclrtDestroyStream(streams[i])
/home/zn/ne

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Multi-NPU execution error

**Link**: https://github.com/ggml-org/llama.cpp/issues/8580
**State**: closed
**Created**: 2024-07-19T02:42:10+00:00
**Closed**: 2024-07-27T08:36:45+00:00
**Comments**: 2
**Labels**: medium severity, Ascend NPU

### Description

### What happened?

When using CANN as the backend, and using more than one npu card. graph split is not correct causing execution to get stuck.

build cmd: cmake .. -DCMAKE_BUILD_TYPE=debug -DLLAMA_CANN=on && make -j
exec cmd: ./bin/llama-cli  -m /root/qwen2-7b-instruct-fp16.gguf  -ngl 32 --split-mode layer --repeat_penalty 1.0 --color -i  -r "User:" -f ../prompts/chat-with-bob.txt

### Name and Version

version: 3408 (1bdd8ae1)
built with cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0 for aarch64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
here's some logs:

llm_load_tensors: ggml ctx size =    0.45 MiB
llm_load_tensors:        CPU buffer size =  1039.50 MiB
llm_load_tensors:       CANN buffer size =  6668.17 MiB   # CANN0
llm_load_tensors:       CANN buffer size =  6818.60 MiB   # CANN1
........................................................................................
llama_new_context_with_model: n_ctx    

[... truncated for brevity ...]

---

