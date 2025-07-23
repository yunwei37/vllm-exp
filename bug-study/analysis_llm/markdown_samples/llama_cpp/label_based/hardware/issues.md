# hardware - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 29

### Label Distribution

- hardware: 30 issues
- build: 8 issues
- enhancement: 8 issues
- bug: 6 issues
- stale: 4 issues
- performance: 4 issues
- need more info: 3 issues
- question: 3 issues
- duplicate: 3 issues
- wontfix: 2 issues

---

## Issue #N/A: segmentation fault Alpaca

**Link**: https://github.com/ggml-org/llama.cpp/issues/317
**State**: closed
**Created**: 2023-03-20T09:56:07+00:00
**Closed**: 2023-04-17T07:12:17+00:00
**Comments**: 35
**Labels**: hardware

### Description

Hello, 
I've tried out the Aplaca model but after a while there comes an error I believe stating: "zsh: segmentation fault  ./main -m ./models/alpaca/ggml-alpaca-7b-q4.bin --color -f  -ins". 
Thanks.

Code: 
./main -m ./models/alpaca/ggml-alpaca-7b-q4.bin --color -f ./prompts/alpaca.txt -ins
main: seed = 1679305614
llama_model_load: loading model from './models/alpaca/ggml-alpaca-7b-q4.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: ggml ctx size = 4529.34 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: loading model part 1/1 from './models/alpaca/ggml-alpaca-7b-q4.bin'
llama_model_load: .................................... don

[... truncated for brevity ...]

---

## Issue #N/A: Running " python3 convert-pth-to-ggml.py models/7B/ 1 " and running out of RAM

**Link**: https://github.com/ggml-org/llama.cpp/issues/200
**State**: closed
**Created**: 2023-03-16T09:01:36+00:00
**Closed**: 2023-03-16T15:04:32+00:00
**Comments**: 8
**Labels**: wontfix, need more info, hardware

### Description

No description provided.

---

## Issue #N/A: Is it possible to run llama.cpp in Google Colab Pro?

**Link**: https://github.com/ggml-org/llama.cpp/issues/128
**State**: closed
**Created**: 2023-03-14T12:38:11+00:00
**Closed**: 2023-03-15T21:27:56+00:00
**Comments**: 2
**Labels**: hardware

### Description

Any help or guidance would be greatly appreciated.

---

## Issue #N/A: FP16 and 4-bit quantized model both produce garbage output on M1 8GB

**Link**: https://github.com/ggml-org/llama.cpp/issues/137
**State**: closed
**Created**: 2023-03-14T17:05:51+00:00
**Closed**: 2023-03-14T20:54:06+00:00
**Comments**: 4
**Labels**: hardware

### Description

Both the `ggml-model-q4_0` and `ggml-model-f16` produce a garbage output on my M1 Air 8GB, using the 7B LLaMA model. I've seen the quantized model having problems but I doubt the quantization is the issue as the non-quantized model produces the same output.

```
➜  llama.cpp git:(master) ./main -m ./models/7B/ggml-model-f16.bin -p "Building a website can be done in 10 simple steps:" -t 8 -n 512
main: seed = 1678812348
llama_model_load: loading model from './models/7B/ggml-model-f16.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 1
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: ggml ctx size = 13365.09 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: loading model part 1/1 from '.

[... truncated for brevity ...]

---

## Issue #N/A: Docker Issus ''Illegal instruction''

**Link**: https://github.com/ggml-org/llama.cpp/issues/537
**State**: closed
**Created**: 2023-03-26T19:18:11+00:00
**Closed**: 2024-04-12T01:07:28+00:00
**Comments**: 24
**Labels**: bug, hardware, stale

### Description

I try to make it run the docker version on Unraid, 

I run this as post Arguments:
`--run -m /models/7B/ggml-model-q4_0.bin -p "This is a test" -n 512`

I got this error:  `/app/.devops/tools.sh: line 40:     7 Illegal instruction     ./main $arg2`

Log:
```
main: seed = 1679843913
llama_model_load: loading model from '/models/7B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: type    = 1
llama_model_load: ggml ctx size = 4273.34 MB
llama_model_load: mem required  = 6065.34 MB (+ 1026.00 MB per state)
/app/.devops/tools.sh: line 40:     7 Illegal instruction     ./main $arg2
```

I have run this whitout any issus:  `--all-in-one "/models

[... truncated for brevity ...]

---

## Issue #N/A: Unable to compile - error: inlining failed in call to ‘always_inline’ ‘_mm256_cvtph_ps’

**Link**: https://github.com/ggml-org/llama.cpp/issues/159
**State**: closed
**Created**: 2023-03-15T10:53:18+00:00
**Closed**: 2023-03-15T15:23:31+00:00
**Comments**: 2
**Labels**: hardware, build

### Description

Hi, I downloaded the files with git and run make just as in the instruction. Unfortunately, the compilation is not working. Can someone help me figure out what's going wrong here?

I'm adding the full error in the following.

``In file included from /usr/lib/gcc/x86_64-linux-gnu/10/include/immintrin.h:113,
                 from ggml.c:155:
ggml.c: In function ‘ggml_vec_dot_f16’:
/usr/lib/gcc/x86_64-linux-gnu/10/include/f16cintrin.h:52:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_cvtph_ps’: target specific option mismatch
   52 | _mm256_cvtph_ps (__m128i __A)
      | ^~~~~~~~~~~~~~~
ggml.c:911:33: note: called from here
  911 | #define GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(x)))
      |                                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ggml.c:921:37: note: in expansion of macro ‘GGML_F32Cx8_LOAD’
  921 | #define GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx8_LOAD(p)
      |                               

[... truncated for brevity ...]

---

## Issue #N/A: RISC-V support?

**Link**: https://github.com/ggml-org/llama.cpp/issues/165
**State**: closed
**Created**: 2023-03-15T16:07:46+00:00
**Closed**: 2023-07-07T13:48:11+00:00
**Comments**: 9
**Labels**: enhancement, hardware

### Description

By deleting line 155 (#include <immintrin.h>) in ggml.c, it works just fine on RISC-V.
Maybe this can be added in Cmake?

---

## Issue #N/A: Can it support avx cpu's older than 10 years old?

**Link**: https://github.com/ggml-org/llama.cpp/issues/451
**State**: closed
**Created**: 2023-03-24T02:19:30+00:00
**Closed**: 2023-07-28T19:40:41+00:00
**Comments**: 10
**Labels**: enhancement, hardware, build

### Description

I can't run any model due to my cpu is from before 2013.So I don't have avx2 instructions.Can you please support avx cpus?

---

## Issue #N/A: .pth to .ggml Out of Memory

**Link**: https://github.com/ggml-org/llama.cpp/issues/76
**State**: closed
**Created**: 2023-03-13T02:56:50+00:00
**Closed**: 2023-03-13T03:05:56+00:00
**Comments**: 2
**Labels**: wontfix, hardware

### Description

I have 16 GBs of memory (14 GB free) and running `python3 convert-pth-to-ggml.py models/7B/ 1` causes an OOM error (Killed) on Linux.

Here's the dmesg message:
`Out of memory: Killed process 930269 (python3) total-vm:15643332kB, anon-rss:13201980kB, file-rss:4kB, shmem-rss:0kB, UID:0 pgtables:26524kB oom_score_adj:0`

I will be receiving my new RAM in a few days but I think this is supposed to work with 16 GB memory?

---

## Issue #N/A: Is it possible to run 65B with 32Gb of Ram ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/503
**State**: closed
**Created**: 2023-03-25T17:17:10+00:00
**Closed**: 2023-03-26T10:18:47+00:00
**Comments**: 6
**Labels**: question, hardware, model

### Description

I already quantized my files with this command ./quantize ./ggml-model-f16.bin.X E:\GPThome\LLaMA\llama.cpp-master-31572d9\models\65B\ggml-model-q4_0.bin.X 2 , the first time it reduced my files size from 15.9 to 4.9Gb and when i tried to do it again nothing changed. After i executed this command "./main -m ./models/65B/ggml-model-q4_0.bin -n 128 --interactive-first" and when everything is loaded i enter my prompt, my memory usage goes to 98% (25Gb by main.exe) and i just wait dozens of minutes with nothing that appears heres an example:

**PS E:\GPThome\LLaMA\llama.cpp-master-31572d9> ./main -m ./models/65B/ggml-model-q4_0.bin -n 128 --interactive-first
main: seed = 1679761762
llama_model_load: loading model from './models/65B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 8192
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 64
llama_model_load: n_layer = 80
llama_model_load: n

[... truncated for brevity ...]

---

## Issue #N/A: mpi : attempt inference of 65B LLaMA on a cluster of Raspberry Pis

**Link**: https://github.com/ggml-org/llama.cpp/issues/2164
**State**: open
**Created**: 2023-07-10T16:12:22+00:00
**Comments**: 54
**Labels**: help wanted, 🦙., hardware, research 🔬

### Description

Now that distributed inference is supported thanks to the work of @evanmiller in #2099 it would be fun to try to utilize it for something cool. One such idea is to connect a bunch of [Raspberry Pis](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) in a local network and run the inference using MPI:

```bash
# sample cluster of 8 devices (replace with actual IP addresses of the devices)
$ cat ./hostfile
192.168.0.1:1
192.168.0.2:1
192.168.0.3:1
192.168.0.4:1
192.168.0.5:1
192.168.0.6:1
192.168.0.7:1
192.168.0.8:1

# build with MPI support
$ make CC=mpicc CXX=mpicxx LLAMA_MPI=1 -j

# run distributed inference over 8 nodes
$ mpirun -hostfile ./hostfile -n 8 ./main -m /mnt/models/65B/ggml-model-q4_0.bin -p "I believe the meaning of life is" -n 64
```

Here we assume that the 65B model data is located on a network share in `/mnt` and that `mmap` works over a network share.
Not sure if that is the case - if not, then it would be more difficult to perform th

[... truncated for brevity ...]

---

## Issue #N/A: ClBlast - no gpu load, no perfomans difference.

**Link**: https://github.com/ggml-org/llama.cpp/issues/1217
**State**: closed
**Created**: 2023-04-28T16:05:41+00:00
**Closed**: 2023-05-05T00:51:53+00:00
**Comments**: 28
**Labels**: performance, hardware, build

### Description

How i build:

1.  I use [w64devkit](https://github.com/skeeto/w64devkit/releases)
2. I download [CLBlast](https://github.com/CNugteren/CLBlast) and [OpenCL-SDK](https://github.com/KhronosGroup/OpenCL-SDK)
3. Put folders lib and include from [CLBlast](https://github.com/CNugteren/CLBlast) and [OpenCL-SDK](https://github.com/KhronosGroup/OpenCL-SDK) to w64devkit_1.18.0\x86_64-w64-mingw32
4. Using w64devkit.exe cd to llama.cpp
5. make LLAMA_CLBLAST=1
6. Put clblast.dll near main.exe

When load i got this: 

> Initializing CLBlast (First Run)...
> Attempting to use: Platform=0, Device=0 (If invalid, program will crash)
> Using Platform: AMD Accelerated Parallel Processing Device: gfx90c
> llama_init_from_file: kv self size  = 1600.00 MB
> 
> system_info: n_threads = 7 / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | VSX = 0 |
> main: interactive mod

[... truncated for brevity ...]

---

## Issue #N/A: RISC-V (TH1520&D1) benchmark and hack for <1GB DDR device

**Link**: https://github.com/ggml-org/llama.cpp/issues/288
**State**: closed
**Created**: 2023-03-19T10:14:34+00:00
**Closed**: 2024-04-10T01:08:06+00:00
**Comments**: 8
**Labels**: enhancement, need more info, hardware, stale

### Description

Hi, 
   Just test on RISC-V board: 
   4xC910 2.0G TH1520 LicheePi4A (https://sipeed.com/licheepi4a)  with 16GB LPDDR4X.
   about 6s/token without any instruction acceleration, and it should be <5s/token when boost to 2.5GHz.

```
llama_model_load: ggml ctx size = 668.34 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: .................................... done
llama_model_load: model size =  4017.27 MB / num tensors = 291

system_info: n_threads = 4 / 4 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | FMA = 0 | NEON = 0 | ARM_FMA = 0 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | VSX = 0 | 

main: prompt: 'They'
main: number of tokens in prompt = 2
     1 -> ''
 15597 -> 'They'

sampling parameters: temp = 0.800000, top_k = 40, top_p = 0.950000, repeat_last_n = 64, repeat_penalty = 1.300000


They are now available for sale at the cost of Rs 20,5

main: mem per token = 14368644 bytes
main:     load time =    91.25 ms
main:   

[... truncated for brevity ...]

---

## Issue #N/A: Not having enough memory just causes a segfault or something

**Link**: https://github.com/ggml-org/llama.cpp/issues/257
**State**: closed
**Created**: 2023-03-18T07:28:43+00:00
**Closed**: 2023-05-06T18:03:16+00:00
**Comments**: 9
**Labels**: bug, duplicate, hardware, model

### Description

So. I'm trying to build with CMake on Windows 11 and the thing just stops after it's done loading the model.

![image](https://user-images.githubusercontent.com/4723091/226091364-64a488a7-ebb5-4c24-9dd0-1cb81378008d.png)

And apparently, this is a segfault.

![Screenshot_20230318_121935](https://user-images.githubusercontent.com/4723091/226091335-afbf2712-d2b8-4b88-9b44-6b6a43d78565.png)

Yay yay yyayy yyayay

this is a memory allocation failure it seems, from me not having enough memory. not like llama.cpp Tells Me That lmao, it just segfaults

(`ctx->mem_buffer` is nullptr which probably means the malloc just failed)

---

## Issue #N/A: fix (perf/UX): get physical cores for Windows

**Link**: https://github.com/ggml-org/llama.cpp/issues/1189
**State**: closed
**Created**: 2023-04-26T14:41:54+00:00
**Closed**: 2024-04-09T01:09:51+00:00
**Comments**: 1
**Labels**: enhancement, hardware, windows, threading, stale

### Description

Complete https://github.com/ggerganov/llama.cpp/pull/934 with the windows impl of physical cores

The impl is approximately: 
```c++
DWORD buffer_size = 0;
DWORD result = GetLogicalProcessorInformation(NULL, &buffer_size);
// assert result == FALSE && GetLastError() == ERROR_INSUFFICIENT_BUFFER
PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(buffer_size);
result = GetLogicalProcessorInformation(buffer, &buffer_size);
if (result != FALSE) {
    int num_physical_cores = 0;
    DWORD_PTR byte_offset = 0;
    while (byte_offset < buffer_size) {
        if (buffer->Relationship == RelationProcessorCore) {
            num_physical_cores++;
        }
        byte_offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        buffer++;
    }
    std::cout << "Number of physical cores: " << num_physical_cores << std::endl;
} else {
    std::cerr << "Error getting logical processor information: " << GetLastError() << std::endl;


[... truncated for brevity ...]

---

## Issue #N/A: [ERROR] Using "make" command

**Link**: https://github.com/ggml-org/llama.cpp/issues/443
**State**: closed
**Created**: 2023-03-23T22:26:52+00:00
**Closed**: 2023-04-22T17:29:32+00:00
**Comments**: 3
**Labels**: hardware, build

### Description

Hello evryone, 

I have an issue when i run "make" cmd : 
I use Ubuntu 22.04 in VirtualBox
Make version : GNU Make 4.3


Here the return of cmd 

<pre>I llama.cpp build info: 

I UNAME_S:  Linux

I UNAME_P:  x86_64

I UNAME_M:  x86_64

I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -msse3

I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread

I LDFLAGS:  

I CC:       cc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0

I CXX:      g++ (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0



cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -msse3   -c ggml.c -o ggml.o

In file included from <b>/usr/lib/gcc/x86_64-linux-gnu/11/include/immintrin.h:101</b>,

                 from <b>ggml.c:158</b>:

<b>ggml.c:</b> In function ‘<b>ggml_vec_dot_f16</b>’:

<b>/usr/lib/gcc/x86_64-linux-gnu/11/include/f16cintrin.h:52:1:</b> <font color="#C01C28"><b>error: </b></font>inlining failed in call to ‘<b>always_inline</

[... truncated for brevity ...]

---

## Issue #N/A: make issue on sbc odroid

**Link**: https://github.com/ggml-org/llama.cpp/issues/482
**State**: closed
**Created**: 2023-03-24T23:32:44+00:00
**Closed**: 2023-05-18T10:54:07+00:00
**Comments**: 2
**Labels**: need more info, hardware

### Description

I am trying to run "make" on an odroid sbc and get following error:

`I llama.cpp build info: 
I UNAME_S:  Linux
I UNAME_P:  unknown
I UNAME_M:  armv7l
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:  
I CC:       cc (Debian 10.2.1-6) 10.2.1 20210110
I CXX:      g++ (Debian 10.2.1-6) 10.2.1 20210110

cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations   -c ggml.c -o ggml.o
ggml.c: In function ‘ggml_vec_mad_q4_0’:
ggml.c:2049:35: warning: implicit declaration of function ‘vzip1_s8’; did you mean ‘vzipq_s8’? [-Wimplicit-function-declaration]
 2049 |             const int8x8_t vxlt = vzip1_s8(vxls, vxhs);
      |                                   ^~~~~~~~
      |                              

[... truncated for brevity ...]

---

## Issue #N/A: CUDA/OpenCL error, out of memory when reload.

**Link**: https://github.com/ggml-org/llama.cpp/issues/1456
**State**: closed
**Created**: 2023-05-14T17:56:02+00:00
**Closed**: 2023-06-09T23:16:05+00:00
**Comments**: 25
**Labels**: bug, high priority, hardware

### Description

Hello folks,

When try `save-load-state` example with CUDA, error occured.
It seems to necessary to add something toward `llama_free` function.

`n_gpu_layers` variable is appended at main function like below.

```cpp
int main(int argc, char ** argv) {
    ...
    auto lparams = llama_context_default_params();

    lparams.n_ctx     = params.n_ctx;
    lparams.n_parts   = params.n_parts;
    lparams.n_gpu_layers = params.n_gpu_layers; // Add gpu layers count
    lparams.seed      = params.seed;
    ...
}
```

And tried to run as below.

```dos
D:\dev\pcbangstudio\workspace\my-llama\bin>save-load-state.exe -m ggml-vic7b-q4_0.bin -ngl 32
main: build = 548 (60f8c36)
llama.cpp: loading model from ggml-vic7b-q4_0.bin
llama_model_load_internal: format     = ggjt v2 (latest)
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 512
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model

[... truncated for brevity ...]

---

## Issue #N/A: make LLAMA_CUBLAS=1 && ./perplexity generates GPU load, while ./main does not

**Link**: https://github.com/ggml-org/llama.cpp/issues/1283
**State**: closed
**Created**: 2023-05-02T16:09:22+00:00
**Closed**: 2023-05-02T16:49:14+00:00
**Comments**: 2
**Labels**: bug, hardware

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Running:

`llama.cpp$ ./main -t 16 -m /data/llama/7B/ggml-model-q4_0.bin -b 512 -p "Building a website can be done in 10 simple steps:" -n 512
`

I believe _should_ generate load on my NVidia 1080Ti, but it doesn't:
```
$ nvidia-smi -i 0
Tue May  2 15:57:44 20

[... truncated for brevity ...]

---

## Issue #N/A: M1 Max + GNU coreutils: "Your arch is announced as x86_64, but it seems to actually be ARM64"

**Link**: https://github.com/ggml-org/llama.cpp/issues/101
**State**: closed
**Created**: 2023-03-13T19:57:53+00:00
**Closed**: 2024-04-10T01:08:09+00:00
**Comments**: 2
**Labels**: bug, hardware, build, stale

### Description

When I build, the makefile detects my M1 Max as 86_64.

This is because I have GNU coreutils `uname` on my `PATH`, which announces my architecture as `arm64` (whereas the system distribution of `uname` would call the same architecture `arm`).

https://github.com/Lightning-AI/lightning/pull/13992#issuecomment-1204157830  
https://github.com/Lightning-AI/lightning/issues/13991

this condition needs widening to accept both `arm` and `arm64`:

https://github.com/ggerganov/llama.cpp/blob/c09a9cfb06c87d114615c105adda91b0e6273b69/Makefile

---

## Issue #N/A: Why Q4 much faster than Q8 ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/1239
**State**: closed
**Created**: 2023-04-29T18:45:38+00:00
**Closed**: 2023-05-12T11:38:42+00:00
**Comments**: 5
**Labels**: performance, hardware

### Description

I've tried to check inference performance for different quantised formats expecting Q8_0 to be fastest due to smaller number of shifts / moves and other CPU operations.

To my surprise it lags behind the Q4_0, which I expected to be slower. 

So I'm curious what's the main reason for that - just the fact that maybe Q8 is not well  supported yet, or Q4 faster due to some fundamental laws, like less moves between RAM <-> CPU, etc?

Is it expected for Q4 to be faster for future releases too?

---

## Issue #N/A: Error: inlining failed in call to always_inline ‘_mm256_cvtph_ps’: target specific option mismatch

**Link**: https://github.com/ggml-org/llama.cpp/issues/107
**State**: closed
**Created**: 2023-03-13T23:20:27+00:00
**Closed**: 2023-03-14T18:08:16+00:00
**Comments**: 22
**Labels**: duplicate, good first issue, hardware, build

### Description

I cloned the GitHub repository and ran the make command but was unable to get the cpp files to compile successfully. Any help or suggestion would be appreciated.

Terminal output:
<pre><font color="#4E9A06"><b>brickman@Ubuntu-brickman</b></font>:<font color="#3465A4"><b>~/Desktop/llama.cpp</b></font>$ ls
CMakeLists.txt  convert-pth-to-ggml.py  ggml.c  ggml.h  LICENSE  main.cpp  Makefile  <font color="#3465A4"><b>models</b></font>  quantize.cpp  <font color="#4E9A06"><b>quantize.sh</b></font>  README.md  utils.cpp  utils.h
<font color="#4E9A06"><b>brickman@Ubuntu-brickman</b></font>:<font color="#3465A4"><b>~/Desktop/llama.cpp</b></font>$ make
I llama.cpp build info: 
I UNAME_S:  Linux
I UNAME_P:  x86_64
I UNAME_M:  x86_64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -msse3
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:  
I CC:       cc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
I CXX:      g++ (Ubuntu 9.4.0-1u

[... truncated for brevity ...]

---

## Issue #N/A: Add avx-512 support?

**Link**: https://github.com/ggml-org/llama.cpp/issues/160
**State**: closed
**Created**: 2023-03-15T12:10:17+00:00
**Closed**: 2023-03-28T09:54:15+00:00
**Comments**: 6
**Labels**: enhancement, performance, hardware

### Description

No clue but I think it may work faster

---

## Issue #N/A: any interest in the openchatkit on a power book? 

**Link**: https://github.com/ggml-org/llama.cpp/issues/96
**State**: closed
**Created**: 2023-03-13T16:43:04+00:00
**Closed**: 2023-07-28T19:30:06+00:00
**Comments**: 9
**Labels**: enhancement, question, hardware

### Description

https://www.together.xyz/blog/openchatkit this new repository might also be a good candidate for any local deployment with a strong GPU. As the gptNeox focus is on GPU deployments.


---

## Issue #N/A: Will there ever be a GPU support for Apple Silicon?

**Link**: https://github.com/ggml-org/llama.cpp/issues/164
**State**: closed
**Created**: 2023-03-15T16:06:51+00:00
**Closed**: 2023-03-15T20:10:04+00:00
**Comments**: 1
**Labels**: enhancement, hardware

### Description

I really thank you for the possibility of running the model on my MacBook Air M1. I've been testing various parameters and I'm happy even with the 7B model. However, do you plan to utilize the GPU of M1/M2 chip? Thank you in advance.

---

## Issue #N/A: making on linuxmint 21

**Link**: https://github.com/ggml-org/llama.cpp/issues/208
**State**: closed
**Created**: 2023-03-16T13:52:27+00:00
**Closed**: 2023-05-06T17:55:19+00:00
**Comments**: 2
**Labels**: duplicate, hardware, build

### Description

im running on bare metal nothing emulated

```
littlemac@littlemac:~$` git clone https://github.com/ggerganov/llama.cpp
Cloning into 'llama.cpp'...
remote: Enumerating objects: 283, done.
remote: Counting objects: 100% (283/283), done.
remote: Compressing objects: 100% (113/113), done.
remote: Total 283 (delta 180), reused 255 (delta 164), pack-reused 0
Receiving objects: 100% (283/283), 158.38 KiB | 609.00 KiB/s, done.
Resolving deltas: 100% (180/180), done.
cd littlemac@littlemac:~$ cd llama.cpp/
littlemac@littlemac:~/llama.cpp$ make
I llama.cpp build info: 
I UNAME_S:  Linux
I UNAME_P:  x86_64
I UNAME_M:  x86_64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -msse3
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:  
I CC:       cc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0
I CXX:      g++ (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0

cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -msse3   -c ggml.c

[... truncated for brevity ...]

---

## Issue #N/A: convert the 7B model to ggml FP16 format fails on RPi 4B

**Link**: https://github.com/ggml-org/llama.cpp/issues/138
**State**: closed
**Created**: 2023-03-14T17:47:38+00:00
**Closed**: 2023-03-15T21:19:53+00:00
**Comments**: 9
**Labels**: hardware

### Description

Everything's OK until this step

python3 convert-pth-to-ggml.py models/7B/ 1
{'dim': 4096, 'multiple_of': 256, 'n_heads': 32, 'n_layers': 32, 'norm_eps': 1e-06, 'vocab_size': 32000}
n_parts =  1
Processing part  0
Killed


models/7B/ggml-model-f16.bin isn't created


---

## Issue #N/A: On the edge llama?

**Link**: https://github.com/ggml-org/llama.cpp/issues/1052
**State**: closed
**Created**: 2023-04-19T01:24:08+00:00
**Closed**: 2023-04-23T12:46:33+00:00
**Comments**: 1
**Labels**: question, hardware

### Description

Sorry to ask this... But is possible to get llama.cpp working on things like edge TPU?

https://coral.ai/products/accelerator-module/

---

## Issue #N/A: Error: inlining failed in call to ‘always_inline’ ‘_mm256_cvtph_ps’ on x86_64 - better support for different x86_64 CPU instruction extensions

**Link**: https://github.com/ggml-org/llama.cpp/issues/196
**State**: closed
**Created**: 2023-03-16T04:17:08+00:00
**Closed**: 2023-03-30T08:31:50+00:00
**Comments**: 35
**Labels**: bug, performance, hardware, build

### Description

When I compile with make, the following error occurs
```
inlining failed in call to ‘always_inline’ ‘_mm256_cvtph_ps’: target specific option mismatch
   52 | _mm256_cvtph_ps (__m128i __A)
```

Error will be reported when executing `cc  -I.   -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -msse3   -c ggml.c -o ggml.o` .
But the error of executing `cc  -I.   -O3 -DNDEBUG -std=c11   -fPIC -pthread  -msse3   -c ggml.c -o ggml.o` will not occur.
Must `-mavx` be used with `-mf16c`?

---
OS: Arch Linux x86_64
Kernel: 6.1.18-1-lts

---

## Issue #N/A: simde?

**Link**: https://github.com/ggml-org/llama.cpp/issues/10
**State**: closed
**Created**: 2023-03-11T11:05:50+00:00
**Closed**: 2023-03-12T06:24:14+00:00
**Comments**: 1
**Labels**: enhancement, hardware

### Description

Could [simde](https://github.com/simd-everywhere/simde) help with porting to x86?

---

