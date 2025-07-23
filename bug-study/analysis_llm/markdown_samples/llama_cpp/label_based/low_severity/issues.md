# low_severity - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- low severity: 30 issues
- bug-unconfirmed: 27 issues
- stale: 16 issues
- Ascend NPU: 2 issues
- Vulkan: 1 issues
- Nvidia GPU: 1 issues
- documentation: 1 issues
- enhancement: 1 issues
- help wanted: 1 issues
- devops: 1 issues

---

## Issue #N/A: Bug: `-ngl` is missing from server docs for layer offload

**Link**: https://github.com/ggml-org/llama.cpp/issues/8605
**State**: closed
**Created**: 2024-07-20T20:45:02+00:00
**Closed**: 2024-09-05T01:07:00+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

There's no mention of how to offload layers to gpu

### Name and Version

docs

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: Bug: There is an issue to execute llama-baby-llama.

**Link**: https://github.com/ggml-org/llama.cpp/issues/9478
**State**: closed
**Created**: 2024-09-14T02:21:32+00:00
**Closed**: 2024-10-01T08:33:00+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

Whenever I go to execute llama-baby-llama, I get “ggml/src/ggml.c:6793: GGML_ASSERT(false && ‘backwards pass not implemented’) failed “ error.

### Name and Version

./llama-baby-llama -m ./models/Qwen-7B-Chat/Qwen-7B-Chat-Q4_0.gguf -p "I believe the meaning of life is" -n 128

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: Bug: Random inputs generated automatically in llama-cli

**Link**: https://github.com/ggml-org/llama.cpp/issues/9456
**State**: closed
**Created**: 2024-09-12T20:07:34+00:00
**Closed**: 2024-11-16T01:59:06+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

I am running the cuurent model : 

 ./llama.cpp/llama-cli -m /home/piuser/Desktop/Abhrant/Meta-Llama-3-8B.Q4_K_S.gguf -n 512 --repeat_penalty 1.0 --color -i -r "User:" -f llama.cpp/prompts/chat-with-bob.txt
 
 When I do this, the cli starts and the conversation goes on normally. Sometimes, a random input is automatically taken even when I am not giving it. 
 For example:
<img width="923" alt="Screenshot 2024-09-13 at 1 35 13 AM" src="https://github.com/user-attachments/assets/8eccdfdd-e625-464b-9f6d-5d8bc85208f5">


I have added the question "what can you do? ". I have not added the input "I love you Bob." it automatically came up after the answer to "what can you do? " was generated. Any idea why? 

 

### Name and Version

version: 3733 (1b280614)
built with cc (Debian 12.2.0-14) 12.2.0 for aarch64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
[1726171239] == Running in interactive mode. =

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Unable to call llama.cpp inference server with llama 3 model 

**Link**: https://github.com/ggml-org/llama.cpp/issues/7978
**State**: closed
**Created**: 2024-06-17T14:19:05+00:00
**Closed**: 2024-08-01T01:07:02+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

After following the installation instructions, I am unable to call the inference server on a llama 3 model. 
<img width="688" alt="image" src="https://github.com/ggerganov/llama.cpp/assets/45519735/97c85173-4e8b-4d4d-9c6e-915d02bb7bb5">



### Name and Version

./llama-server --model ~/Project/src/models/llama-3-neural-chat-v1-8b-Q5_K_M.gguf --port 8080  -cb --version
version: 3166 (21be9cab)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
./llama-server --model ~/Project/src/models/llama-3-neural-chat-v1-8b-Q5_K_M.g
guf --port 8080  -cb 
INFO [                    main] build info | tid="140605182199744" timestamp=1718633396 build=3166 commit="21be9cab"
INFO [                    main] system info | tid="140605182199744" timestamp=1718633396 n_threads=4 n_threads_batch=-1 total_threads=8 system_info="AVX = 1 | AVX_VNNI = 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: JSON Schema-to-GBNF additionalProperties bugs (and other minor quirks)

**Link**: https://github.com/ggml-org/llama.cpp/issues/7789
**State**: closed
**Created**: 2024-06-06T05:26:46+00:00
**Closed**: 2024-07-12T21:00:19+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

While debugging json-schema-to-gbnf grammars, I noticed a few bugs / quirks and wanted to write them down somewhere.

# `additionalProperties` seems to default to `false` (not matching spec). 

By default, additional properties [should be permitted](https://json-schema.org/understanding-json-schema/reference/object#properties). However, providing a schema like:

```json
{
  "type": "object",
  "properties": {
    "number": { "type": "number" },
    "street_name": { "type": "string" },
    "street_type": { "enum": ["Street", "Avenue", "Boulevard"] }
  }
}
```
Then it correctly passes on these strings:
```json
{ "number": 1600, "street_name": "Pennsylvania", "street_type":"Avenue"}
```
```json
{ "street_name": "Pennsylvania" }
```
```json
{ "number": 1600, "street_name": "Pennsylvania" }
```
```json
{}
```

But then it improperly fails on the string:
```json
{ "number": 1600, "street_name": "Pennsylvania", "street_type":"Avenue", "dir

[... truncated for brevity ...]

---

## Issue #N/A: Bug: ImportError: libprotobuf-lite.so.25: cannot open shared object file: No such file or directory

**Link**: https://github.com/ggml-org/llama.cpp/issues/9071
**State**: closed
**Created**: 2024-08-18T08:36:11+00:00
**Closed**: 2024-08-19T13:49:16+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

Arch has a newer protobuf that apparently is not compatible with llama.cpp's llava. I tried building an older version of protobuf but was unsuccessful. 

### Name and Version

> ./llama-cli --version
version: 3600 (2fb92678)
built with clang version 17.0.0 for x86_64-pc-linux-gnu


### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
python ./examples/llava/convert_image_encoder_to_gguf.py -m /thearray/git/models/dolphin-vision-72b/vit --llava-projector /thearray/git/models/dolphin-vision-72b/vit/llava.projector --output-dir /thearray/git/models/dolphin-vision-72b/vit/ --clip-model-is-vision
Traceback (most recent call last):
  File "/code/git/llama.cpp/./examples/llava/convert_image_encoder_to_gguf.py", line 8, in <module>
    from gguf import *
  File "/code/git/llama.cpp/gguf-py/gguf/__init__.py", line 7, in <module>
    from .vocab import *
  File "/code/git/llama.cpp/gguf-py/gguf/vocab.py", line 10,

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Or Feature? BPE Tokenization mutates whitespaces into double-whitespace tokens when add_prefix_space is true (default)

**Link**: https://github.com/ggml-org/llama.cpp/issues/8023
**State**: closed
**Created**: 2024-06-20T01:51:25+00:00
**Closed**: 2024-09-13T01:07:22+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

This is a bit discussed here already: https://github.com/ggerganov/llama.cpp/issues/7938
`<|assistant|> `
```
32001 -> '<|assistant|>'
   259 -> '  '
```

Also `<|assistant|>\n`:
```
32001 -> '<|assistant|>'
29871 -> ' '
    13 -> '
'
```

What happens is that the single whitespace, that follows a special token is mutated into a double-whitespace token (259) because add_prefix_space is triggered in llama.cpp when a special token is encountered.

In the second example the template actually wants a \n after assistant, however the special behavior sneaks a space in between.

Is this intended behavior / correct ?

When running PHI3 and asking for a generation after `<|assistant|>`, phi3 is adamant in responding with a whitespace or a combination token that starts with a whitespace. 
When disabling add_prefix_whitespace and adding a `\n` after assistant, this issue is resolved and phi responds right away with normal text.


### Name and Version

ba

[... truncated for brevity ...]

---

## Issue #N/A: Bug: RPC server doesn't load GPU if I use Vulkan 

**Link**: https://github.com/ggml-org/llama.cpp/issues/8536
**State**: closed
**Created**: 2024-07-17T09:08:36+00:00
**Closed**: 2024-10-03T10:00:53+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

I compiled llamacpp with Vulkan backend. The "rpc-server" binary is linked to libvulkan but it never uses my GPUs. While "llama-cli" is OK.

### Name and Version

version: 3384 (4e24cffd)
built with cc (GCC) 14.1.1 20240701 (Red Hat 14.1.1-7) for x86_64-redhat-linux

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
./rpc-server
create_backend: using CPU backend
Starting RPC server on 0.0.0.0:50052, backend memory: 23967 MB


ldd ./rpc-server
        linux-vdso.so.1 (0x00007f18759f2000)
        libllama.so => /home/metal3d/Projects/ML/llama.cpp/build-rpc/src/libllama.so (0x00007f1875879000)
        libggml.so => /home/metal3d/Projects/ML/llama.cpp/build-rpc/ggml/src/libggml.so (0x00007f1875400000)
        libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007f1875000000)
        libm.so.6 => /lib64/libm.so.6 (0x00007f187531c000)
        libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007f187582b000)
 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: RuntimeError: Internal: could not parse ModelProto from ../llama3/Meta-Llama-3-8B-Instruct/tokenizer.model

**Link**: https://github.com/ggml-org/llama.cpp/issues/8484
**State**: closed
**Created**: 2024-07-15T04:27:06+00:00
**Closed**: 2024-08-29T01:07:04+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

How to slove this question
RuntimeError: Internal: could not parse ModelProto from ../llama3/Meta-Llama-3-8B-Instruct/tokenizer.model

### Name and Version

llama3 install and quantize the data

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Bug: MESA: error: ../src/intel/vulkan/anv_device.c:4237: VK_ERROR_OUT_OF_DEVICE_MEMORY

**Link**: https://github.com/ggml-org/llama.cpp/issues/8492
**State**: closed
**Created**: 2024-07-15T12:04:53+00:00
**Closed**: 2024-09-05T01:07:06+00:00
**Comments**: 7
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

I have an Intel ARC750 graphic card.  The same Phi-3-mini-4k-instruct-fp16.gguf can be run on x86 host with vulkan backend successfully, but it failed on RISC-V host

### Name and Version

./llama-cli --version
version: 3372 (a977c115)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for riscv64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
root@Ubuntu-riscv64:~/liyong/llama.cpp/build/bin# ./llama-cli -m ../../../../Phi-3-mini-4k-instruct-fp16.gguf -p "Hi you how are you" -n 50 -e -ngl 33 -t 4
Log start
main: build = 3372 (a977c115)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for riscv64-linux-gnu
main: seed  = 1721069901
llama_model_loader: loaded meta data with 23 key-value pairs and 195 tensors from ../../../../Phi-3-mini-4k-instruct-fp16.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this ou

[... truncated for brevity ...]

---

## Issue #N/A: Bug: prefix completion endpoint with /v1

**Link**: https://github.com/ggml-org/llama.cpp/issues/7740
**State**: closed
**Created**: 2024-06-04T12:53:18+00:00
**Closed**: 2024-06-04T13:05:00+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

The server should respect OpenAPI path, but "completion" endpoint is not prefixed by "/v1"

This is required by NeoVim LLM-LS plugin which uses "openapi" backend to call "/v1/completion".

Thanks

### Name and Version

version: 2902 (9afdffe7)
built with cc (GCC) 14.0.1 20240411 (Red Hat 14.0.1-0) for x86_64-redhat-linux


### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: The image generated by dockerfile cannot be used

**Link**: https://github.com/ggml-org/llama.cpp/issues/7987
**State**: closed
**Created**: 2024-06-18T02:54:02+00:00
**Closed**: 2024-08-02T01:20:40+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

docker build -t local/llama.cpp:llama-server -f .devops/llama-server.Dockerfile .
error message 
./server: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by ./server)

The problem did not occur after adding the following line
apt install build-essential



### Name and Version

$./llama-server 

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
The problem did not occur after adding the following line
```


---

## Issue #N/A: Bug: Build fails on i386 systems

**Link**: https://github.com/ggml-org/llama.cpp/issues/9545
**State**: closed
**Created**: 2024-09-19T03:09:15+00:00
**Closed**: 2024-11-04T01:07:30+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, Vulkan, stale, low severity

### Description

### What happened?

```
/wrkdirs/usr/ports/misc/llama-cpp/work/llama.cpp-b3761/ggml/src/ggml-vulkan.cpp:2629:5: error: no matching function for call to 'vkCmdCopyBuffer'
 2629 |     vkCmdCopyBuffer(subctx->s->buffer, staging->buffer, dst->buffer, 1, &buf_copy);
      |     ^~~~~~~~~~~~~~~
/usr/local/include/vulkan/vulkan_core.h:4750:28: note: candidate function not viable: no known conversion from 'vk::Buffer' to 'VkBuffer' (aka 'unsigned long long') for 2nd argument
 4750 | VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(
      |                            ^
 4751 |     VkCommandBuffer                             commandBuffer,
 4752 |     VkBuffer                                    srcBuffer,
      |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Version: 3761
clang-18
FreeBSD 14.1

### Name and Version

3761

### What operating system are you seeing the problem on?

BSD

### Relevant log output

_No response_

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

## Issue #N/A: Bug: [Hardware: ppc64le] On ppc64le llama.cpp only uses 1 thread by default and not half of all threads as it does on x86

**Link**: https://github.com/ggml-org/llama.cpp/issues/9623
**State**: closed
**Created**: 2024-09-24T07:32:59+00:00
**Closed**: 2024-11-09T01:07:03+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

I'm having an 8 core Power10 system with SMT=2 (=16 threads), but only 1 of the 16 threads is used by default.

When I run a sample prompt like:

```bash
export MODELS=gemma-2-2b-it-q4_k_m.gguf
./build/bin/llama-cli -m ${MODELS} -p '10 simple steps to build a website'
```

it only uses 1/16 threads as you can see in the output. This can be fixed by starting with `-t` parameter but ideally it should take half of the cores as it does on Intel/x86 HW.

<img width="558" alt="image" src="https://github.com/user-attachments/assets/ed3f5cce-e286-4b26-bfa3-7e0c5df877fb">


### Name and Version

```bash
# Llama version
$ ./build/bin/llama-cli --version
version: 3818 (31ac5834)
built with cc (GCC) 13.1.1 20230614 (Red Hat 13.1.1-4) for ppc64le-redhat-linux

# OS
$ cat /etc/os-release 
NAME="AlmaLinux"
VERSION="9.3 (Shamrock Pampas Cat)"
ID="almalinux"
ID_LIKE="rhel centos fedora"
VERSION_ID="9.3"
PLATFORM_ID="platform:el9"
PRETTY_NAME="AlmaLinux 9.3 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Inconsistency while parsing the model using `llama-cli` and `gguf-py`

**Link**: https://github.com/ggml-org/llama.cpp/issues/9893
**State**: closed
**Created**: 2024-10-15T04:49:25+00:00
**Closed**: 2024-11-29T01:09:54+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

Hi, recently, I'm trying to learn the gguf-py lib and use the gruff-py and write a script to make a gguf file, after I made the file, I tried to load it using llama-cli, but it said I have the wrong tensor number. So I'm wondering if there are some inconsistencies between the cpp loader and the py loader.

Here, my script is:
``` python
import os
import re
import ast
import sys
import random
import uuid
import string
import subprocess
from pathlib import Path

import numpy as np

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFWriter

writer = GGUFWriter('test.gguf', 'llama')

model='llama'
token_l=11
context_len=123
emb_len=234
bc=1
ff_len=345
hc=10
rms_eps=0.1
tokenizer_model='llama'

token_list = random.sample(string.printable, token_l)
writer.add_token_list(token_list)
writer.add_context_length(context_len)
writer.add_embedding_length(emb_len)
writer.add_b

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Can't make imatrix quant of Q4_0_X_X

**Link**: https://github.com/ggml-org/llama.cpp/issues/9190
**State**: closed
**Created**: 2024-08-26T14:40:36+00:00
**Closed**: 2024-08-26T17:44:44+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

When trying to quantize Q4_0_4_4 (and others) with imatrix, I get errors about `GGML_ASSERT(result == nrows * row_size)`

### Name and Version

b3615 ubuntu 22.04

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
[   1/ 543]                    token_embd.weight - [ 7168, 64000,     1,     1], type =    f16,
| ====== llama_model_quantize_internal: did not find weights for token_embd.weight
converting to q4_0 .. size =   875.00 MiB ->   246.09 MiB
[   2/ 543]               blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[   3/ 543]                blk.0.ffn_down.weight - [20480,  7168,     1,     1], type =    f16, converting to q4_0_4x4 .. ggml/src/ggml.c:20598: GGML_ASSERT(result == nrows * row_size) failed
ggml/src/ggml.c:20598: GGML_ASSERT(result == nrows * row_size) failed
ggml/src/ggml.c:20598: GGML_ASSERT(result == nrows * row_size) failed
ggml/src/ggml.c:20598: GGM

[... truncated for brevity ...]

---

## Issue #N/A: llama.cpp is slow on GPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/9881
**State**: closed
**Created**: 2024-10-14T11:19:53+00:00
**Closed**: 2024-12-01T01:08:05+00:00
**Comments**: 9
**Labels**: Nvidia GPU, bug-unconfirmed, stale, low severity

### Description

### What happened?

llama.cpp is running slow on NVIDIA A100 80GB GPU

Steps to reproduce:
1. git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
2. mkdir build && cd build
3. cmake .. -DGGML_CUDA=ON
4. make GGML_CUDA=1
3. command:  ./build/bin/llama-cli -m ../gguf_files/llama-3-8B.gguf -t 6912 --ctx-size 50 --n_predict 50 --prompt "There are two persons named ram and krishna"
 Here threads are set to 6912 since GPU has 6912 CUDA cores.

It is slow on gpu compared to cpu.
On gpu  eval time is around 0.07 tokens per second.
Is this expected behaviour or any tweak should be done while building llama.cpp?

### Name and Version

version: 3902 (c81f3bbb)
built with cc (GCC) 11.4.1 20231218 (Red Hat 11.4.1-3) for aarch64-redhat-linux

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
```


---

## Issue #N/A: Bug: Vulkan, I-quants partially working since PR #6210 (very slow, only with all repeating layers offloaded)

**Link**: https://github.com/ggml-org/llama.cpp/issues/7976
**State**: closed
**Created**: 2024-06-17T13:11:03+00:00
**Closed**: 2024-06-17T14:51:44+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

I-quants suddenly started working on Vulkan backend after #6210 was merged, albeit at very slow speeds (token generation is even slowr than when using a single cpu thread). 

But, it only works if at least all layers exept the last one (every "repeating layers") are oflloaded to GPU. Anything else (even `-ngl 0`) and it crashes with `GGML_ASSERT: C:\[...]\llama.cpp\ggml-vulkan.cpp:3006: d_X->size >= x_sz * ne02 * ne03`

## Example llama-bench outputs: 

### Vulkan (q6-k):
ggml_vulkan: Found 1 Vulkan devices:
Vulkan0: AMD Radeon RX 5700 XT (AMD proprietary driver) | uma: 0 | fp16: 1 | warp size: 64
| model                          |       size |     params | backend    | ngl | threads | n_batch |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | ------------: | ---------------: |
| llama 1B Q6_K                  | 860.87 MiB |     1.10 B | Vulkan     |  23 |       6 |   

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Version infomation missing when using llama-cli built on mac

**Link**: https://github.com/ggml-org/llama.cpp/issues/9977
**State**: closed
**Created**: 2024-10-21T09:06:53+00:00
**Closed**: 2024-12-06T01:07:37+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

Hi! After failing to run the pre-compiled binaries on my Mac, I tried to compile them myself. 
However, after finishing the compilation, I found the version information missing when I ran the `llama-cli --version` command. (The compiled binary can infer the "Hello world" prompt via phi-3 model normally.)


### Name and Version

./llama.cpp-b3938/build_metal/bin/llama-cli --version
version: 0 (unknown)
built with Apple clang version 14.0.0 (clang-1400.0.29.202) for arm64-apple-darwin21.6.0

### What operating system are you seeing the problem on?

Mac

### Relevant log output

_No response_

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

## Issue #N/A: Refactor: Add CONTRIBUTING.md and/or update PR template with [no ci] tips

**Link**: https://github.com/ggml-org/llama.cpp/issues/7657
**State**: closed
**Created**: 2024-05-30T23:56:20+00:00
**Closed**: 2024-06-09T15:25:57+00:00
**Comments**: 4
**Labels**: documentation, enhancement, help wanted, devops, low severity

### Description

### Background Description

Discussion in https://github.com/ggerganov/llama.cpp/pull/7650 pointed out a need to add a CONTRIBUTING.md and maybe add a PR template to encourage contributors to add [no ci] tag to documentation only changes.

https://docs.github.com/en/actions/managing-workflow-runs/skipping-workflow-runs

(If anyone wants to tackle this, feel free to)

### Possible Refactor Approaches

Add info about

- doc only changes should have [no ci] in commit title to skip the unneeded CI checks.
- squash on merge with commit title format: "module : some commit title (`#1234`)"

---

## Issue #N/A: Bug: Unable to enable AVX_VNNI instructions

**Link**: https://github.com/ggml-org/llama.cpp/issues/10116
**State**: closed
**Created**: 2024-11-01T02:14:40+00:00
**Closed**: 2024-11-05T01:24:49+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

I have built it on Windows env with OneAPI setup and try to enable AVX_VNNI instructions. My computer is support AVX_VNNI:
![image](https://github.com/user-attachments/assets/7c513c5f-1353-4468-81b7-5dcbd8d209fd)
and I followed the [doc's steps](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#intel-onemkl) :

```shell
# in OneAPI env, execute:
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_NATIVE=ON
cmake --build build --config Release
```

but it still shows no AVX_VNNI support:
![image](https://github.com/user-attachments/assets/974f6b65-a80b-47a0-a059-0b076c19c768)

#### my platform
+ msvc: MSVC 19.41.34123.0( Visual Studio 2022) on Windows SDK 10.0.22631.
+ oneAPI:  oneAPI 2024.2


### Name and Version

.\llama-cli.exe --version
version: 3983 (8841ce3f)
built with MSVC 19.41.34123.0 for x64

### What operating system are you seeing the problem on?

Windows



[... truncated for brevity ...]

---

## Issue #N/A: Bug: --log-disable also disables output from the model

**Link**: https://github.com/ggml-org/llama.cpp/issues/10155
**State**: closed
**Created**: 2024-11-04T03:06:11+00:00
**Closed**: 2024-12-11T02:04:50+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

When using any client command (i.e llama-cli llama-llava-cli) with the <code>--log-disable</code> or the equivalent <code>-lv -1</code> option, the model's response to the prompt is not printed.

This seems to be unexpected behavior, as the model's response is not considered a log - nor is it output via stderr in any mode. It should be noted that this also occurs in conversation-mode, even the prompt-character (>) is not printed. Furthermore, in normal-mode, the process eventually reports itself as `Killed`.

The expected behavior, is that the model's response to the prompt (or conversation) is printed <strong>but the debug information (i.e llama_model_loader: - kv ...) is not</strong>.

### Name and Version

version: 4016 (42cadc74)
built with clang version 18.1.8+libcxx for x86_64-pc-linux-musl

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
% llama-cli --log-disable -m ./models/8B/favorite-8B

[... truncated for brevity ...]

---

## Issue #N/A: Bug: convert-hf-to-gguf.py on Gemma model ValueError: Duplicated key name 'tokenizer.chat_template'

**Link**: https://github.com/ggml-org/llama.cpp/issues/7923
**State**: closed
**Created**: 2024-06-13T19:09:23+00:00
**Closed**: 2024-07-21T01:53:03+00:00
**Comments**: 3
**Labels**: bug, low severity

### Description

### What happened?

When trying to convert

https://huggingface.co/SakanaAI/DiscoPOP-zephyr-7b-gemma/

I get the error in the title, but it's only defined a single time in tokenizer_config.json:

https://huggingface.co/SakanaAI/DiscoPOP-zephyr-7b-gemma/blob/main/tokenizer_config.json#L59

Verified locally with `cat *.json | grep chat_template` and I only get the one result

Is it somehow trying to load it twice?

Looks like when Gemma is initialized, it runs _set_vocab_sentencepiece(), which runs special_vocab.add_to_gguf (which pulls in the chat_template), and then it also again runs special_vocab.add_to_gguf

but that would mean it's been broken since April 16..

https://github.com/ggerganov/llama.cpp/pull/6689

### Name and Version

b3145 ubuntu 22.04

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
INFO:hf-to-gguf:Loading model: DiscoPOP-zephyr-7b-gemma
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endia

[... truncated for brevity ...]

---

## Issue #N/A: Bug: llama-quantize --help is not printed

**Link**: https://github.com/ggml-org/llama.cpp/issues/10122
**State**: closed
**Created**: 2024-11-01T12:29:32+00:00
**Closed**: 2024-12-22T01:07:42+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

Appended `--help` does not print help immediately, but starts quantization or throws error:
```shell
./llama-quantize model-bf16.gguf --help IQ4_NL
./llama-quantize model-bf16.gguf IQ4_NL --help
```

### Name and Version

c02e5ab2a675c8bc1abc8b1e4cb6a93b26bdcce7

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
./build/bin/llama-quantize --leave-output-tensor ./qwen2.5-coder-7b-instruct/qwen2.5-coder-7B-instruct-BF16.gguf --help IQ
4_NL                                                                                                                                                          
main: build = 4000 (c02e5ab2)                                                                                                                                 
main: built with cc (Ubuntu 13.2.0-23ubuntu4) 13.2.0 for x86_64-linux-gnu                                                                                     
main: qua

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Name Error when running Llava1.5 examples

**Link**: https://github.com/ggml-org/llama.cpp/issues/10190
**State**: closed
**Created**: 2024-11-06T03:48:13+00:00
**Closed**: 2024-11-08T11:37:01+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

when running examples Llava1.5, 

python ./examples/llava/convert_image_encoder_to_gguf.py -m ../clip-vit-large-patch14-336 --llava-projector ../llava-v1.5-7b/llava.projector --output-dir ../llava-v1.5-7b

the process raise an error:
NameError: name 'KEY_EMBEDDING_LENGTH' is not defined

which happens in convert_image_encoder_to_gguf.py line 204:
fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), v_hparams["hidden_size"])

### Name and Version

version: 4020 (9f409893)
built with Apple clang version 13.0.0 (clang-1300.0.29.30) for x86_64-apple-darwin23.6.0

### What operating system are you seeing the problem on?

Mac

### Relevant log output

_No response_

---

## Issue #N/A: Getting bash: ./perplexity: No such file or directory 

**Link**: https://github.com/ggml-org/llama.cpp/issues/8431
**State**: closed
**Created**: 2024-07-11T07:44:20+00:00
**Closed**: 2024-08-10T11:57:41+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

Getting the bash: ./perplexity: No such file or directory error even after the Build has happened correctly 
Is the tool renamed?

### Name and Version

Lasted version

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: Bug: image encoding error with malloc memory

**Link**: https://github.com/ggml-org/llama.cpp/issues/10225
**State**: closed
**Created**: 2024-11-09T02:01:05+00:00
**Closed**: 2024-11-15T02:11:32+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

When compiling the Minicpm model, during the process of image encoding, memory allocation operations are performed for the variables.
The first subgraph execution runs fine, but during the second malloc, it reports a core dump error. I have not been able to identify the cause. Please help me.



code file: llava.cpp
code:  image_embd_v[i] = (float *)malloc(clip_embd_nbytes(ctx_clip));  ##clip_embd_nbytes(ctx_clip):917504




if (clip_is_minicpmv(ctx_clip)) {
        std::vector<float *> image_embd_v;
        image_embd_v.resize(img_res_v.size);
        struct clip_image_size * load_image_size = clip_image_size_init();
        for (size_t i = 0; i < img_res_v.size; i++) {
            const int64_t t_img_enc_step_start_us = ggml_time_us();
            image_embd_v[i] = (float *)malloc(clip_embd_nbytes(ctx_clip));
            int patch_size=14;
            load_image_size->width = img_res_v.data[i].nx;
            load_image_size->height = img_res_v

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Vulkan backend not work on an Imagination GPU on RISC-V Platform

**Link**: https://github.com/ggml-org/llama.cpp/issues/8437
**State**: closed
**Created**: 2024-07-11T11:01:33+00:00
**Closed**: 2024-08-26T01:06:58+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

Nothing output after "Vulkan0: PowerVR B-Series BXE-2-32 (PowerVR B-Series Vulkan Driver) | uma: 1 | fp16: 1 | warp size: 1"
It is on a RISC-V board with an imagination igpu 

### Name and Version

./llama-cli --version version: 3369 (278d0e18) built with cc (Ubuntu 13.2.0-4ubuntu3-bb2) 13.2.0 for riscv64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
root@k1:~/liyong/llama.cpp/build/bin# ./llama-cli -m ../../../Phi-3-mini-4k-instruct-fp16.gguf -p "Hi you how are you" -n 50 -e -ngl 33 -t 4
Log start
main: build = 3369 (278d0e18)
main: built with cc (Ubuntu 13.2.0-4ubuntu3-bb2) 13.2.0 for riscv64-linux-gnu
main: seed  = 1720706611
llama_model_loader: loaded meta data with 23 key-value pairs and 195 tensors from ../../../Phi-3-mini-4k-instruct-fp16.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_m

[... truncated for brevity ...]

---

