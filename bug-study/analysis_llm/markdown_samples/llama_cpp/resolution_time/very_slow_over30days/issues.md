# very_slow_over30days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- stale: 26 issues
- bug-unconfirmed: 14 issues
- enhancement: 7 issues
- low severity: 2 issues
- bug: 2 issues
- research 🔬: 2 issues
- high severity: 1 issues
- help wanted: 1 issues
- good first issue: 1 issues
- windows: 1 issues

---

## Issue #N/A: Bug: When llama-server.exe or llama-cli.exe is executed without any error messages appearing, it may not be operating normally when GGUF models are enabled.

**Link**: https://github.com/ggml-org/llama.cpp/issues/8616
**State**: closed
**Created**: 2024-07-21T16:23:24+00:00
**Closed**: 2024-09-04T01:07:02+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

After executing a command in cmd, it only shows a window process in the task tray, but there is no window displayed on the desktop, no errors, nothing happens, I don't know what's going on.

![chrome_Xn7J2VSY2T](https://github.com/user-attachments/assets/f5ef9447-5795-4333-a502-cff9e404be11)

cmd
```
"C:\Tools\AI_Translator_Tools\llama-b3426-bin-win-cuda-cu12.2.0-x64\llama-server.exe" -m causallm_14b.Q8_0.gguf -c 2048
```


### Name and Version

OS: WIN10 Enterprise 21H2 LTSC
Version: [llama-b3426-bin-win-cuda-cu12.2.0-x64.zip](https://github.com/ggerganov/llama.cpp/releases/download/b3426/llama-b3426-bin-win-cuda-cu12.2.0-x64.zip) or [llama-b3428-bin-win-cuda-cu12.2.0-x64.zip](https://github.com/ggerganov/llama.cpp/releases/download/b3428/llama-b3428-bin-win-cuda-cu12.2.0-x64.zip)

### What operating system are you seeing the problem on?

Windows

### Relevant log output


No information was reported, it just ran a command and then stopped. 

[... truncated for brevity ...]

---

## Issue #N/A: Windows 64-bit, Microsoft Visual Studio - it works like a charm after those fixes!

**Link**: https://github.com/ggml-org/llama.cpp/issues/22
**State**: closed
**Created**: 2023-03-11T20:44:33+00:00
**Closed**: 2023-04-16T10:25:54+00:00
**Comments**: 40
**Labels**: enhancement, help wanted, good first issue, windows

### Description

First of all thremendous work Georgi! I managed to run your project with a small adjustments on:
- Intel(R) Core(TM) i7-10700T CPU @ 2.00GHz / 16GB as x64 bit app, it takes around 5GB of RAM.

<img width="622" alt="image" src="https://user-images.githubusercontent.com/95347171/224509962-6ed8d954-66bc-4531-8dd0-423cc2ee5e2c.png">

<img width="568" alt="image" src="https://user-images.githubusercontent.com/95347171/224510066-a8adccfa-d9db-4546-8efb-e69efc549b97.png">

Here is the list of those small fixes:

- main.cpp: added ggml_time_init() at start of main (division by zero otherwise)
- quantize.cpp: same as above at start of main (division by zero otherwise)
- ggml.c: #define QK 32 moved to dedicated define.h (should not be in .c)
- ggml.c: replace fopen with fopen_s (VS secure error message)
- ggml.c: below changes due to 'expression must be a pointer or complete object type':
1. 2x `(uint8_t*)(y` to: `((uint8_t*)y` 
2. 4x `(const uint8_t*)(x` to `((const uint8_t*)x`


[... truncated for brevity ...]

---

## Issue #N/A: CUDA/HIP stream usage on ROCm causes constant 100% GPU load - support disabling streams on ROCm

**Link**: https://github.com/ggml-org/llama.cpp/issues/3929
**State**: closed
**Created**: 2023-11-03T11:33:21+00:00
**Closed**: 2024-04-02T01:12:21+00:00
**Comments**: 7
**Labels**: enhancement, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

The CUDA/ROCm implementation in llama.cpp uses CUDA (HIP) streams for multi-GPU support - by default, 8 per GPU. It's fairly easy in the code to reduce this to 1, but there's no provision for disabling stream usage altogether. Unfortunately on ROCm, at least with RD

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: llama3 and 3.1 uncensored

**Link**: https://github.com/ggml-org/llama.cpp/issues/8895
**State**: closed
**Created**: 2024-08-06T20:52:47+00:00
**Closed**: 2024-09-24T01:07:22+00:00
**Comments**: 8
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF
https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored
Add 70b also

### Motivation

Helps to get uncensored answers

### Possible Implementation

_No response_

---

## Issue #N/A: Eval bug: llama-mtmd-cli doesn't support system prompts

**Link**: https://github.com/ggml-org/llama.cpp/issues/13454
**State**: closed
**Created**: 2025-05-11T14:18:24+00:00
**Closed**: 2025-07-12T01:08:26+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

```
$ bin/llama-mtmd-cli --version
version: 5343 (62d4250e)
built with cc (GCC) 15.1.1 20250425 (Red Hat 15.1.1-1) for x86_64-redhat-linux
```

### Operating systems

Linux

### GGML backends

HIP

### Hardware

Ryzen 9 5950X + AMD Radeon RX 6700 XT

### Models

Bug does not depend on hardware or model

### Problem description & steps to reproduce

When running llama-mtmd-cli with --help, the documentation for --prompt suggests -sys is an available argument:
```
-p,    --prompt PROMPT                  prompt to start generation with; for system message, use -sys
```

However, attempting to run with -sys passed will result in an error.

### First Bad Commit

_No response_

### Relevant log output

```shell
error: invalid argument: -sys
error: invalid argument: --system-prompt
```

---

## Issue #N/A: llama.cpp BPE tokenization of wiki.test does not match the HF tokenization

**Link**: https://github.com/ggml-org/llama.cpp/issues/3502
**State**: closed
**Created**: 2023-10-06T13:42:05+00:00
**Closed**: 2024-04-06T01:06:25+00:00
**Comments**: 10
**Labels**: stale

### Description

I did the following test to tokenize `wiki.test.raw` using our tokenizer and the Python tokenizer.
The expectation is that the outputs will match:

```bash
# generate ggml-vocab-falcon.gguf
./convert-falcon-hf-to-gguf.py --vocab-only ~/development/huggingface/falcon-7b/ --outfile ./models/ggml-vocab-falcon.gguf

# tokenize using Python
python3 tests/test-tokenizer-0-falcon.py ~/development/huggingface/falcon-7b/ --fname-tok ./build/wikitext-2-raw/wiki.test.raw

# tokenize using llama.cpp
cd build
make -j
./bin/test-tokenizer-0-falcon ../models/ggml-vocab-falcon.gguf ./wikitext-2-raw/wiki.test.raw

# compare the results
cmp ./wikitext-2-raw/wiki.test.raw.tok ./wikitext-2-raw/wiki.test.raw.tokcpp 
./wikitext-2-raw/wiki.test.raw.tok ./wikitext-2-raw/wiki.test.raw.tokcpp differ: char 1, line 1
```

The results are pretty close, but not exactly the same. Any ideas why the test does not pass?
I thought that #3252 would resolve this

cc @goerch 

---

## Issue #N/A: Misc. bug: test-backend-ops grad crash by GGML_ASSERT error

**Link**: https://github.com/ggml-org/llama.cpp/issues/12520
**State**: closed
**Created**: 2025-03-22T23:56:35+00:00
**Closed**: 2025-05-06T01:07:43+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

.\llama-cli.exe --version
version: 4942 (fbdfefe7)
built with MSVC 19.43.34808.0 for x64

### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

Test code

### Command line

```shell
> .\test-backend-ops.exe grad -o CPY
or
> .\test-backend-ops.exe grad
```

### Problem description & steps to reproduce

## description

Commit #12310 crashes test-backend-ops grad.
It doesn't seem to matter which backend.

## steps to reproduce

Run `test-backend-ops` as `grad` mode.

### First Bad Commit

Commit #12310 : SHA ba932dfb50cc694645b1a148c72f8c06ee080b17

### Relevant log output

```shell
[3/23 08:24:26] PS E:\AI\llama.cpp\b4942\llama-b4942-bin-win-vulkan-x64
> .\test-backend-ops.exe grad -o CPY
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon(TM) Graphics (AMD proprietary driver) | uma: 1 | fp16: 1 | warp size: 64 | shared memory: 32768 | matrix cores: none
Testing 2 devices

Backend 1/2: Vulkan0
  Device description: AMD

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Empty response in interactive mode or incomplete answer

**Link**: https://github.com/ggml-org/llama.cpp/issues/1133
**State**: closed
**Created**: 2023-04-22T22:16:11+00:00
**Closed**: 2024-04-09T01:09:57+00:00
**Comments**: 3
**Labels**: stale

### Description

When running with the -t 12 -i -r "### Human:" flags llama returns control
the cpu activity goes to 0 and the user sends a new input
however, llama now continues responding to the previous input (or returns no response) completely ignoring the new input...

from here llama completely breaks the chat, it even generates the prompt "### Human:" and starts completing the questions written by a human

Using the --ignore-eos does not help, llama keeps stopping randomly.

Note: tested this many times with vicuna and gpt4all
Note: this might be related to issues https://github.com/ggerganov/llama.cpp/discussions/990 https://github.com/ggerganov/llama.cpp/issues/941 https://github.com/ggerganov/llama.cpp/discussions/993

---

## Issue #N/A: [LoRA] Support safetensors lora

**Link**: https://github.com/ggml-org/llama.cpp/issues/3714
**State**: closed
**Created**: 2023-10-21T13:27:55+00:00
**Closed**: 2024-04-04T01:07:35+00:00
**Comments**: 1
**Labels**: stale

### Description

Airoboros released their last PEFT as safetensors. The conversion script appears to mainly support .bin unless I'm wrong. I'm sure others will release loras in that format in the future.

---

## Issue #N/A: main : failed to eval

**Link**: https://github.com/ggml-org/llama.cpp/issues/5727
**State**: closed
**Created**: 2024-02-26T09:18:26+00:00
**Closed**: 2024-04-12T01:06:42+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

b2267 main
When I use main.exe ,I constantly input information into the model, he is broken! 
but server.exe not.

here is the message:
-------------------------------------------------------------------------------------------------
'''bash
D:\soul>main.exe -m qwen-1.8b-q4_0.gguf -ngl 99 -ins
Log start
main: build = 2267 (c3937339)
main: built with MSVC 19.37.32822.0 for x64
main: seed  = 1708938405
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6, VMM: yes
llama_model_loader: loaded meta data with 19 key-value pairs and 195 tensors from qwen-1.8b-q4_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen
llama_model_loader: - kv   1:              

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: Converting the Model to Llama.cpp GGUF

**Link**: https://github.com/ggml-org/llama.cpp/issues/10969
**State**: closed
**Created**: 2024-12-24T19:08:26+00:00
**Closed**: 2025-02-21T01:07:21+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale

### Description

### Git commit

https://github.com/ggerganov/llama.cpp/releases/tag/b4390

### Operating systems

Windows

### GGML backends

CPU

### Problem description & steps to reproduce

https://www.datacamp.com/tutorial/llama3-fine-tuning-locally

I am trying this code in Kaggle Notebook. In this tutorial, I tried to "3. Converting the Model to Llama.cpp GGUF". But I had some issues. Could you help me about these issues ? 


%cd /kaggle/working
!git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
%cd /kaggle/working/llama.cpp
!sed -i 's|MK_LDFLAGS   += -lcuda|MK_LDFLAGS   += -L/usr/local/nvidia/lib64 -lcuda|' Makefile
!LLAMA_CUDA=1 conda run -n base make -j > /dev/null

/kaggle/working
Cloning into 'llama.cpp'...
remote: Enumerating objects: 1217, done.
remote: Counting objects: 100% (1217/1217), done.
remote: Compressing objects: 100% (944/944), done.
remote: Total 1217 (delta 260), reused 765 (delta 221), pack-reused 0 (from 0)
Receiving objects: 100% (1217/1217), 

[... truncated for brevity ...]

---

## Issue #N/A: Better description for flags related to prompt template

**Link**: https://github.com/ggml-org/llama.cpp/issues/4413
**State**: closed
**Created**: 2023-12-11T20:04:15+00:00
**Closed**: 2024-04-03T01:14:21+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

I'm trying to format the Llama-2 prompt, but I couldn't exactly get it to work. It would be great if manual has better description for some of the flags and examples.

From manual:

* -i: run in interactive mode
* -ins: run in instruction mode (use with Alpaca models)
* -cml: run in chatml mode (use with ChatML-compatible models)
* -p STRING: prompt to start generation with (default: empty)
* --in-prefix-bos: prefix BOS to user inputs, preceding the `--in-prefix` string
* --in-prefix STRING: string to prefix user inputs with (default: empty)
* --in-suffix STRING: string to suffix after user inputs with (default: empty)
* -r STRING: halt generation at PROMPT, return control in interactive mode

My attempt is below, but it doesn't exactly match the prompt because first turn with the system message and the second turn without the system message are slightly different.

Llama-2 prompt: <s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message1} [/INST] {bot_message1} </

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

## Issue #N/A: Batch size affects model's output

**Link**: https://github.com/ggml-org/llama.cpp/issues/249
**State**: closed
**Created**: 2023-03-18T01:03:42+00:00
**Closed**: 2023-07-28T19:34:07+00:00
**Comments**: 10
**Labels**: bug, generation quality

### Description

I was tinkering with the code and made the following change in `line 977, main.cpp` (as it seemed wrong to me):
*from*
```C
if (embd.size() > params.n_batch) {
       break;
}
```
*to*
```C
if (embd.size() >= params.n_batch) {
       break;
}
```

The model's (13B) outputs suddenly changed. Reverted changes and tried to play with the `batch_size` parameter, it really does affect the output.

Not sure if it's expected behaviour. As far as I understand it shouldn't be the case. A bug? Different batch sizes have different evaluation results (rounding error)?

---

## Issue #N/A: How to convert Microsoft/trocr to ggml format

**Link**: https://github.com/ggml-org/llama.cpp/issues/7453
**State**: closed
**Created**: 2024-05-22T07:12:59+00:00
**Closed**: 2024-07-06T01:06:32+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

[microsoft/trocr-small-handwritten](https://huggingface.co/microsoft/trocr-small-handwritten/tree/main)
What method can be used to convert [pytorch_model.bin](https://huggingface.co/microsoft/trocr-small-handwritten/blob/main/pytorch_model.bin) into the traditional ggml format?
thank



---

## Issue #N/A: [FEATURE REQUEST] - "Dithering" to improve quantization results at inference time

**Link**: https://github.com/ggml-org/llama.cpp/issues/4976
**State**: closed
**Created**: 2024-01-16T13:32:03+00:00
**Closed**: 2024-04-03T01:13:47+00:00
**Comments**: 2
**Labels**: research 🔬, stale

### Description

![image](https://github.com/ggerganov/llama.cpp/assets/66376113/7ea87fb4-d236-47b2-9501-80542be3a52c)

In image processing / Audio processing, dithering is used to reduce the perceived impact of quantization error. They intentionally apply noise to "smoothen out" the error caused by quantization.

For images specifically, this is used to reduce visual artifacts like color banding, but dithering techniques have not yet been applied to quantization for neural network weights at scale.

I have tried modifying q4_0 dequantization logic to apply some randomization to see if it would change the outcome probabilities:

```
q4_0, regular inference

Token 1: 88.095337%
Token 2: 3.622236%
Token 3: 1.774177%
Token 4: 0.670536%
Token 5: 0.665275%

q4_0, experimental dequantization logic, same model file

Token 1: 89.557686%
Token 2: 2.744859%
Token 3: 1.819543%
Token 4: 0.582869%
Token 5: 0.545077%

fp16:

Token 1: 91.897667%
Token 2: 2.790743%
Token 3: 0.972574%
Toke

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

## Issue #N/A: Bug: Slow response times with llama.cpp llama-server

**Link**: https://github.com/ggml-org/llama.cpp/issues/9013
**State**: closed
**Created**: 2024-08-13T02:09:32+00:00
**Closed**: 2024-09-28T01:08:19+00:00
**Comments**: 9
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

When running:
.\llama-cli -m gemma-2-2b-it-Q4_K_M.gguf --threads 16 -ngl 27 --mlock --port 11484 --host 0.0.0.0 --top_k 40 --repeat_penalty 1.1 --min_p 0.05 --top_p 0.95 --prompt-cache-all -cb -np 4 --batch-size 512 -cnv

The output is blazing fast. When I sent "write a story" these are my speed stats:
llama_print_timings:        load time =    1027.95 ms
llama_print_timings:      sample time =     683.92 ms /   618 runs   (    1.11 ms per token,   903.62 tokens per second)
llama_print_timings: prompt eval time =    3678.86 ms /    12 tokens (  306.57 ms per token,     3.26 tokens per second)
llama_print_timings:        eval time =    4744.65 ms /   617 runs   (    7.69 ms per token,   130.04 tokens per second)
llama_print_timings:       total time =   15385.66 ms /   629 tokens

When I try the same with llama server:
.\llama-server -m gemma-2-2b-it-Q4_K_M.gguf --threads 16 -ngl 27 --mlock --port 11484 --host 0.0.0.0 --top_k 40 --repeat_penalty 1.1 --mi

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: LLaVa convert_image_encoder_to_gguf.py fails to byteswap v.head.ffn_up.bias tensor on Big-Endian system

**Link**: https://github.com/ggml-org/llama.cpp/issues/12863
**State**: closed
**Created**: 2025-04-10T07:21:30+00:00
**Closed**: 2025-06-06T01:07:58+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

Bug only specific to Python code. Not C/C++ code.

$ git rev-parse HEAD
fe5b78c89670b2f37ecb216306bed3e677b49d9f

### Operating systems

Linux

### GGML backends

CPU, BLAS

### Hardware

IBM z15 8 IFLs / 64 GB RAIM / 160 GB + 500 GB DASD / NOSMT / LPAR

### Models

IBM Granite Vision 3.2 2B F16 (mmproj-model-f16.gguf)

### Problem description & steps to reproduce

### The Problem

Using the following machines for this test:
1. MacBook Air M3 (Little-Endian byte-order)
2. IBM z15 Mainframe (Big-Endian byte-order)

**Steps to reproduce:**
1. On both machines, pull the latest code and follow the (README-granitevision.md)[https://github.com/ggml-org/llama.cpp/blob/master/examples/llava/README-granitevision.md] instructions.
2. On both machines, create the `mmproj-model-f16.gguf` file using the following command

```sh
python3 /opt/llama-testbed/examples/llava/convert_image_encoder_to_gguf.py \
  -m $ENCODER_PATH/ \
  --llava-projector $ENCODER_PATH/llava.projector \


[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: NUMA-aware MoE Expert Allocation for Improved Performanc

**Link**: https://github.com/ggml-org/llama.cpp/issues/11333
**State**: closed
**Created**: 2025-01-21T16:44:28+00:00
**Closed**: 2025-05-26T01:08:10+00:00
**Comments**: 6
**Labels**: enhancement, stale

### Description

### Feature Description

Current llama.cpp implementation doesn't optimally utilize NUMA architecture when running Mixture-of-Experts (MoE) models, potentially leaving significant performance gains untapped. 

### Proposed Solution  
Implement NUMA-aware expert allocation through one or more of these approaches:  
1. **Process-Level Binding**  
   - Integrate `numactl`-like functionality directly into llama.cpp  
   - Allow specifying NUMA nodes per expert group via CLI/config  

2. **Thread Affinity Control**  
   - Add pthread/OpenMP affinity binding for expert computation threads  
   - Example: `--numa-expert-map "0-7:0,8-15:1"` (experts 0-7 on NUMA0, 8-15 on NUMA1)  

3. **NUMA-Aware Memory Allocation**  
   - Leverage `libnuma` for expert weight allocations  
   - Implement `mmap` strategy with `MAP_FIXED_NOREPLACE` for specific nodes  

### Performance Considerations  
- Cross-NUMA communication cost vs. compute density tradeoff  
- Automatic topology detection vs. manual mappin

[... truncated for brevity ...]

---

## Issue #N/A: [User] main program built by cmake crashed due to Illegal instruction

**Link**: https://github.com/ggml-org/llama.cpp/issues/3339
**State**: closed
**Created**: 2023-09-26T12:42:27+00:00
**Closed**: 2024-04-03T01:15:41+00:00
**Comments**: 3
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

cmake built program can be executed without error

# Current Behavior

currenlty program built make can be executed, but main program built by cmake crashed due to 


:~/code/aiu/llama.cpp/build/bin> ./main

Illegal instruction (core dumped)

# Environment 

[... truncated for brevity ...]

---

## Issue #N/A: Cmake file always assumes AVX2 support

**Link**: https://github.com/ggml-org/llama.cpp/issues/1583
**State**: closed
**Created**: 2023-05-24T03:47:38+00:00
**Closed**: 2024-04-09T01:08:52+00:00
**Comments**: 6
**Labels**: bug, high priority, build, stale

### Description

When running `cmake` the default configuration sets AVX2 to be ON even when the current cpu does not support it.
AVX vs AVX2 is handled correctly in the plain makefile.

For cmake, the AVX2 has to be turned off via `cmake -DLLAMA_AVX2=off .` for the compiled binary to work on AVX-only system.

Can we make the cmake file smarter about whether to enable or disable AVX2 by looking at the current architecture?

---

## Issue #N/A: csm : implement Sesame-based conversation example

**Link**: https://github.com/ggml-org/llama.cpp/issues/12392
**State**: closed
**Created**: 2025-03-14T14:49:46+00:00
**Closed**: 2025-05-14T01:07:48+00:00
**Comments**: 23
**Labels**: model, research 🔬, stale, tts

### Description

With the first Sesame CSM model [openly available](https://github.com/SesameAILabs/csm), we should implement a local example similar to their [online research demo](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice#demo). It seems that the released CSM model uses [Kyutai's Mimi](https://arxiv.org/abs/2410.00037) audio codec which we have to implement in a similar way as we did with the [WavTokenizer](https://github.com/ggml-org/llama.cpp/pull/10784). Next we can modify the [talk-llama](https://github.com/ggerganov/whisper.cpp/tree/master/examples/talk-llama) example to support audio generation with the CSM. This way we will be able to plug any LLM for the text response generation and use Sesame for speech input/output.

---

## Issue #N/A: Eval bug: Segmentation fault on vanilla Ubuntu, only with version post 4460, solution: usr/local/lib/libllama.so needs replacing by hand

**Link**: https://github.com/ggml-org/llama.cpp/issues/11451
**State**: closed
**Created**: 2025-01-27T11:53:57+00:00
**Closed**: 2025-03-13T01:07:49+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

Used to work until at least: 
```
llama-cli --version 
version: 4460 (ba8a1f9c)
built with Ubuntu clang version 14.0.0-1ubuntu1.1 for x86_64-pc-linux-gnu
```

Fails with the freshly compiled: 
```
$ build/bin/llama-cli --version 
version: 4564 (acd38efe)
built with Ubuntu clang version 14.0.0-1ubuntu1.1 for x86_64-pc-linux-gnu
```

No build errors whatsoever: 
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- Including CPU backend
-- x86 detected
-- Adding CPU backend variant ggml-cpu: -march=native 
-- Configuring done (0.4s)
-- Generating done (6.6s)
-- Build files have been written to: .../Downloads/llama.cpp/build
... 
```

### Operating systems

Linux

### GGML backends

CPU

### Hardware

USB stick based OS, on e.g. 
```
                                                              
                                       

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Implement CodeGenForCausalLM

**Link**: https://github.com/ggml-org/llama.cpp/issues/11789
**State**: closed
**Created**: 2025-02-10T10:54:52+00:00
**Closed**: 2025-04-07T01:09:07+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Please implement `CodeGenForCausalLM` (Salesforce's models) model support for `convert_hf_to_gguf.py`.

### Motivation

Reason: users may use models like Salesforce/codegen-350M-multi.

### Possible Implementation

_No response_

---

## Issue #N/A: llama.cpp with mistral-7b-instruct-v0.2.Q5_K_M.gguf performance comparison between Intel CPU, nVIDIA GPU and Apple M1/M2

**Link**: https://github.com/ggml-org/llama.cpp/issues/5619
**State**: closed
**Created**: 2024-02-21T01:51:56+00:00
**Closed**: 2024-05-04T01:06:37+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, stale

### Description

On Intel CPU, 8 tokens/s
On Apple M1 and M2 (10 core GPU), 20 tokens/s
On 8 x nVIDIA Quadro P6000, compute capability 6.1, 40 tokens/s

I'd expect 8 nVIDIA GPUs would be at least 8 times faster? Is this expected or am I doing something wrong?

Here's the console output from the server example:
```
./server --host 0.0.0.0 -ngl 33
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 8 CUDA devices:
  Device 0: Quadro P6000, compute capability 6.1, VMM: yes
  Device 1: Quadro P6000, compute capability 6.1, VMM: yes
  Device 2: Quadro P6000, compute capability 6.1, VMM: yes
  Device 3: Quadro P6000, compute capability 6.1, VMM: yes
  Device 4: Quadro P6000, compute capability 6.1, VMM: yes
  Device 5: Quadro P6000, compute capability 6.1, VMM: yes
  Device 6: Quadro P6000, compute capability 6.1, VMM: yes
  Device 7: Quadro P6000, compute capability 6.1, VMM: yes
{"timestamp":1708479356,"level":"INFO","function

[... truncated for brevity ...]

---

## Issue #N/A: [User] chat-with-bob.txt mentions incorrect city

**Link**: https://github.com/ggml-org/llama.cpp/issues/683
**State**: closed
**Created**: 2023-04-01T15:19:42+00:00
**Closed**: 2023-05-03T18:45:35+00:00
**Comments**: 4

### Description

prompts/chat-with-bob.txt mentions that Moscow is the biggest city in Europe, while it is actually Istanbul :)

---

## Issue #N/A: common: Gibberish results and/or crashes due to incorrect character encodings

**Link**: https://github.com/ggml-org/llama.cpp/issues/6396
**State**: closed
**Created**: 2024-03-30T09:32:12+00:00
**Closed**: 2024-05-28T02:13:07+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale

### Description

As of ~~b2579~~ b2646, prompts (among other parameters) are internally stored as `std::string`s, which is basically glorified `std::vector<char>` and do not care or handle character encodings. This will not cause any problem since (as far as I can tell) llama.cpp treats all strings as in UTF-8, but care must be taken when taking strings from external sources.

For example, when parsing command-line arguments, `--prompt` (and maybe other arguments) gets stored directly as `params.prompt`:

https://github.com/ggerganov/llama.cpp/blob/c342d070c64a1ffe35d22c1b16b672e684a30297/common/common.cpp#L215-L222

This (somehow) works on Linux, but thanks to Windows' infinite wisdom `argv` is in ANSI codepage encoding, and will cause gibberish results or a crash to happen soon after since all other parts are expecting a UTF-8 string:

https://github.com/ggerganov/llama.cpp/blob/c342d070c64a1ffe35d22c1b16b672e684a30297/llama.cpp#L10974

https://github.com/ggerganov/llama.cpp/blob/c342d070c6

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Multiple PPL calculation methods

**Link**: https://github.com/ggml-org/llama.cpp/issues/8809
**State**: closed
**Created**: 2024-08-01T10:04:40+00:00
**Closed**: 2024-09-15T01:07:31+00:00
**Comments**: 3
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Hope llama.cpp can support multiple PPL calculation methods.

### Motivation

Llama.cpp is an excellent project that can be used for edge deployment of various models. However, when deploying the quantitative models proposed in some papers, it is difficult to evaluate whether the deployment is successful because the ppl calculation method adopted by llama.cpp is different from the calculation method of the code provided in the paper. Therefore, it is hoped that llama.cpp can support multiple PPL 

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: llama.cpp CPU bound while inferencing against DeepSeek-R1 GGUF

**Link**: https://github.com/ggml-org/llama.cpp/issues/11635
**State**: closed
**Created**: 2025-02-03T23:17:41+00:00
**Closed**: 2025-04-12T01:07:44+00:00
**Comments**: 11
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

$ ./build/bin/llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA L40S, compute capability 8.9, VMM: yes
version: 4625 (5598f475)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

Intel(R) Xeon(R) w5-3425 + NVIDIA L40S

### Models

unsloth/DeepSeek-R1-GGUF

### Problem description & steps to reproduce

When attempting to use llama-cli to inference, it becomes CPU bound and is painfully slow (less than one token per second). nvtop shows that the GPU is 0% utilized (all CPU being used) despite 14 layers and 44GB offloaded to VRAM. I'm following [the instructions outlined on Unsloth's blog](https://unsloth.ai/blog/deepseekr1-dynamic) and running the following command:
`!build/bin/llama-cli \
    --model DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001

[... truncated for brevity ...]

---

