# windows - issues

**Total Issues**: 10
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 10

### Label Distribution

- windows: 10 issues
- bug: 4 issues
- stale: 4 issues
- enhancement: 3 issues
- build: 2 issues
- good first issue: 2 issues
- hardware: 1 issues
- threading: 1 issues
- performance: 1 issues
- documentation: 1 issues

---

## Issue #N/A: w64devkit build segfaults at 0xFFFFFFFFFFFFFFFF

**Link**: https://github.com/ggml-org/llama.cpp/issues/2922
**State**: closed
**Created**: 2023-08-31T04:42:28+00:00
**Closed**: 2023-09-01T13:53:15+00:00
**Comments**: 5
**Labels**: bug, windows

### Description

Steps to reproduce:
1. Install latest w64devkit
2. Build with `make LLAMA_DEBUG=1`
3. Simply run `./main`, regardless of whether you have a model in the default location (I don't)

50% of the time, it will fail. I cannot reproduce it if I build with MSYS2's mingw-w64 toolchain instead.

I bisected it to commit 0c44427df10ee024b4e7ef7bfec56e993daff1db, which adds -march=native to CXXFLAGS.

~~If cv2pdb is to be trusted~~ (confirmed below), the crash happens here:
https://github.com/ggerganov/llama.cpp/blob/8afe2280009ecbfc9de2c93b8f41283dc810609a/common/common.cpp#L723

Something is going wrong before that function call:
```
    llama_model * model  = llama_load_model_from_file(params.model.c_str(), lparams);
00007FF7CBE59448  mov         rax,qword ptr [params]  
00007FF7CBE5944F  add         rax,0C8h  
00007FF7CBE59455  mov         rcx,rax  
00007FF7CBE59458  call        _M_range_check+0F70h (07FF7CBEA0340h)  
00007FF7CBE5945D  mov         rcx,rax  
00007FF7CBE59460

[... truncated for brevity ...]

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

## Issue #N/A: cuBLAS - windows - static not compiling

**Link**: https://github.com/ggml-org/llama.cpp/issues/1092
**State**: closed
**Created**: 2023-04-20T21:28:05+00:00
**Closed**: 2024-04-09T01:10:04+00:00
**Comments**: 5
**Labels**: bug, build, windows, stale

### Description

When static linking is selected the CUDA::cublas_static target is not found.
Dynamic binary compilation works.

---

## Issue #N/A: Windows page fault disk i/o slow on first load

**Link**: https://github.com/ggml-org/llama.cpp/issues/705
**State**: closed
**Created**: 2023-04-02T10:04:24+00:00
**Closed**: 2024-04-11T01:07:14+00:00
**Comments**: 37
**Labels**: performance, windows, stale

### Description

Hello,

As of https://github.com/ggerganov/llama.cpp/pull/613 I have experienced significant regression in model loading speed (I'm on windows, compiled msvc llama.cpp, llama.cpp is located on HDD to prevent SSD wear in my case)

It takes roughly 15 minutes for model to load first time after each computer restart/hibernation, during this time my HDD usage is at 100% and my non-llama.cpp read/write operations are slowed down on my pc
![hdd](https://user-images.githubusercontent.com/76458234/229345728-b597023b-f7e3-4a8b-b550-3159863ba03d.png)

Before that, previous commits took 60 - 180 seconds at worst to load model first time, and after first loading occured, model loaded within 5 - 10 seconds on each program restart until pc reboot/hibernation

Before Commit:
![timings2](https://user-images.githubusercontent.com/76458234/229347345-2053d645-0f26-42ef-9f8e-5fc69ad04e1c.png)

After:
![timings1](https://user-images.githubusercontent.com/76458234/229345966-ee606c92-e7cb-42f6-8

[... truncated for brevity ...]

---

## Issue #N/A: Unable to enter Chinese prompt

**Link**: https://github.com/ggml-org/llama.cpp/issues/646
**State**: closed
**Created**: 2023-03-31T06:43:06+00:00
**Closed**: 2023-04-09T08:03:44+00:00
**Comments**: 8
**Labels**: windows

### Description

Hi!My use is compiled under Windows main.exe, when I type Chinese Prompt, I found that the model seems to be unable to understand, under debugging found that std::getline(std::cin,line) get is empty lines, then I tried Japanese, are the same result.
(Since I am a native Chinese speaker, this question was translated by DeepL)
![image](https://user-images.githubusercontent.com/18028414/229043234-a47c0569-07e1-4731-85d9-121f9774fdc9.png)


---

## Issue #N/A: When running in PowerShell in windows, it works, but throws an error in interactive mode

**Link**: https://github.com/ggml-org/llama.cpp/issues/601
**State**: closed
**Created**: 2023-03-29T17:23:13+00:00
**Closed**: 2023-05-18T10:49:11+00:00
**Comments**: 2
**Labels**: bug, windows

### Description

I built llama.cpp using cmake and then Visual Studio (after many trials and tribulations since I'm pretty new to this), but finally got it working.

Using the 7B model the outputs are reasonable, but when I put the -i tag, it runs, then I hit Ctrl+C, it allows me to enter text, but when I hit enter an error pops up in a windows shown below:

![image](https://user-images.githubusercontent.com/65059714/228617990-0da94e0c-5df4-4311-9d41-0ed5c060df0f.png)

I'm running this on my windows machine, but I have been using WSL to get some stuff to work.

Here's an example of it failing:

`(base) PS G:\llama\llama.cpp> .\bin\Debug\main.exe -m ..\LLaMA\7B\ggml-model-q4_0.bin -i -n 124 -t 24`

`(base) PS G:\llama\llama.cpp> .\bin\Debug\main.exe -m ..\LLaMA\7B\ggml-model-q4_0.bin -i -n 124 -t 24
main: seed = 1680110536
llama_model_load: loading model from '..\LLaMA\7B\ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model

[... truncated for brevity ...]

---

## Issue #N/A: Build your windows binaries with Clang and not MSVC.

**Link**: https://github.com/ggml-org/llama.cpp/issues/534
**State**: closed
**Created**: 2023-03-26T17:12:25+00:00
**Closed**: 2024-04-12T01:07:30+00:00
**Comments**: 5
**Labels**: enhancement, build, windows, stale

### Description

Hello,

Your [windows binaries releases](https://github.com/ggerganov/llama.cpp/releases) have probably been built with MSVC and I think there's a better way to do it.

# Expected Behavior

I have a Intel® Core™ i7-10700K and the builds are supposed to recognize those architectures: [AVX | AVX2 | FMA | SSE3 | F16C]

# Current Behavior

Windows (MSVC build)
```
system_info: n_threads = 14 / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 0 | 
NEON = 0 | ARM_FMA = 0 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | VSX = 0 |
```
It misses the FMA, SSE3 and the F16C architectures.

# Fix with Clang

If you build with Clang you'll get all the architectures right:

Windows (Clang build)
```
system_info: n_threads = 14 / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | 
NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 |
```

# How to build with Clang

1. Install Clang
To do this, you have to install some

[... truncated for brevity ...]

---

## Issue #N/A: How to build on windows?

**Link**: https://github.com/ggml-org/llama.cpp/issues/103
**State**: closed
**Created**: 2023-03-13T20:13:14+00:00
**Closed**: 2023-07-28T19:20:41+00:00
**Comments**: 22
**Labels**: documentation, good first issue, windows

### Description

Please give instructions. There is nothing in README but it says that it supports it 

---

## Issue #N/A: llama.exe doesn't handle relative file paths in Windows correctly

**Link**: https://github.com/ggml-org/llama.cpp/issues/46
**State**: closed
**Created**: 2023-03-12T11:13:54+00:00
**Closed**: 2023-04-16T09:20:58+00:00
**Comments**: 10
**Labels**: bug, model, windows

### Description

Please include the `ggml-model-q4_0.bin` model to actually run the code:

```
% make -j && ./main -m ./models/7B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -t 8 -n 512
I llama.cpp build info: 
I UNAME_S:  Darwin
I UNAME_P:  arm
I UNAME_M:  arm64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -DGGML_USE_ACCELERATE
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:   -framework Accelerate
I CC:       Apple clang version 14.0.0 (clang-1400.0.29.202)
I CXX:      Apple clang version 14.0.0 (clang-1400.0.29.202)

cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -DGGML_USE_ACCELERATE   -c ggml.c -o ggml.o
c++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread -c utils.cpp -o utils.o
c++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread main.cpp ggml.o utils.o -o main  -framework Accelerate
c++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread quantize.cpp ggml.o utils

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

