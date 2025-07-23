# medium_severity - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 29

### Label Distribution

- medium severity: 30 issues
- bug-unconfirmed: 26 issues
- stale: 12 issues
- bug: 3 issues
- model: 1 issues
- Ascend NPU: 1 issues
- need more info: 1 issues

---

## Issue #N/A: Bug: issue in CUDA flash attention

**Link**: https://github.com/ggml-org/llama.cpp/issues/10031
**State**: closed
**Created**: 2024-10-24T08:08:06+00:00
**Closed**: 2024-10-24T11:11:31+00:00
**Comments**: 7
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

There seems to be some memory corruption issue in the CUDA flash attention kernel. 

This is demonstrated by the debug prints inserted before and after the `vec_dot_KQ` device function here:
https://github.com/ggerganov/llama.cpp/compare/master...agray3:llama.cpp:ag_demonstrate_fattn_memory_issue

These print out the first element of Q_ds, which is const in the function so shouldn't be altered. (`Q_ds` is in local memory so it also shouldn't be altered by any other thread.)

However we get the result: 
Before vec_dot_KQ: Q_ds=-32752.000000
After vec_dot_KQ: Q_ds=nan

Q_ds is being altered and becoming NAN. This is reproducible across different GPUs and models. 

### Name and Version

version: 3964 (3488adf3)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: Llama-Quantize : Layers quantized in the wrong order, thus damaging the variable bits tensor quants scheme consistency.

**Link**: https://github.com/ggml-org/llama.cpp/issues/9005
**State**: closed
**Created**: 2024-08-12T12:59:04+00:00
**Closed**: 2024-09-27T01:07:21+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

On master b3573, when quantizing Gemma 9b it:

The tensors are quantized in a wrong order.

Right now, because of the layer jump from 7 to 10 without the ffns of layer 7 to be quantized, it breaks not only the layer quantization order, but also the correlation between ffn_down Q6_K and attn_v Q6_K : From layer 7, some layers will have ffn_down Q6_K and attn_v Q5_K, and some others ffn_down Q5_K and attn_v Q6_K.
This gives us suboptimal quants per BPW.

I expect the tensors to be quantized in the right order.

This, so the Q5_K_M quant, as well as the othersusing "use_more_bits(i_layer, n_layer)" to have a variable quant of ffn_down in conjunction with "use_more_bits(qs.i_attention_wv, qs.n_attention_wv))" to have a variable quant of attn_v.weight, can be optimal.

### Name and Version

main: build = 3573 (2589292c)
main: built with MSVC 19.29.30154.0 for x64

### What operating system are you seeing the problem on?

Windows

### Relevant log ou

[... truncated for brevity ...]

---

## Issue #N/A: Unable to convert a fireworks ai model to GGUF with gguf-my-repo

**Link**: https://github.com/ggml-org/llama.cpp/issues/8451
**State**: closed
**Created**: 2024-07-12T09:14:13+00:00
**Closed**: 2024-07-23T10:53:50+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

I  downloaded one of my models from fireworks.ai and pushed it up into huggingface - you can find it here: [llama-3-8b-instruct-danish](https://huggingface.co/HeRksTAn/llama-3-8B-Instruct-Danish)

I then tried  [gguf-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) in order to convert it to gguf. 


1. using https://huggingface.co/spaces/ggml-org/gguf-my-repo 
2. logged into my account
3. search the hub id for the repository I want converted into gguf, which is HeRksTAn/llama-3-8B-Instruct-Danish
4. I chose Q4_K_M
5. I clicked submit


I get the following error 

`Error: Error converting to fp16: b'INFO:hf-to-gguf:Loading model: llama-3-8B-Instruct-Danish\nTraceback (most recent call last):\n File "/home/user/app/llama.cpp/convert_hf_to_gguf.py", line 3551, in \n main()\n File "/home/user/app/llama.cpp/convert_hf_to_gguf.py", line 3517, in main\n hparams = Model.load_hparams(dir_model)\n File "/home/user/app/llama.cpp/convert_hf_to_gguf.

[... truncated for brevity ...]

---

## Issue #N/A: Bug: src/llama.cpp:15099: Deepseek2 does not support K-shift

**Link**: https://github.com/ggml-org/llama.cpp/issues/8862
**State**: closed
**Created**: 2024-08-05T04:14:23+00:00
**Closed**: 2024-10-30T01:19:54+00:00
**Comments**: 10
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

Hi, when stress testing llama-server (--parallel 3, prompt="Count 1 to 10000 in words") and running deepseek-coder-v2:16b-lite-instruct-q8_0  i got this assertion error in the logs and everything stopped working, so i have to restart llm-server.

**Startup script:**

~/llama.cpp/llama-server -m /usr/share/ollama/.ollama/models/blobs/sha256-373dcfc92e01372709b6164fc836f677a6280e25e9eac5c434c64223207bfc4f --port 8000 --host 0.0.0.0 -ngl 28 -c 24600 --threads 16 --parallel 3 --log-format text --predict -2 --logdir ~/llama.cpp/logs --log-append   $1 $2 >> ~/llama.cpp/logs/deepseek.log 2>&1





### Name and Version

version: 3509 (ecf6b7f2)
built with cc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-22.0.1) for x86_64-redhat-linux


### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
**Logs just before it crashed:**

INFO [   launch_slot_with_task] slot is processing task | tid="139873529581568" timestamp=17228302

[... truncated for brevity ...]

---

## Issue #N/A: Qwen2-57B-A14B-Instruct not supported

**Link**: https://github.com/ggml-org/llama.cpp/issues/7813
**State**: closed
**Created**: 2024-06-07T08:47:04+00:00
**Closed**: 2024-06-07T10:00:28+00:00
**Comments**: 13
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

The converted model fails to load due to unexpected expert tensor dimensions, the current qwen2moe implementation expects it to be `n_ff`/`n_expert_used`, which it is not.

### Name and Version

./main --version
version: 3066 (e141ce62)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
llama_model_load: error loading model: check_tensor_dims: tensor 'blk.0.ffn_gate_exps.weight' has wrong shape; expected  3584,  2368,    64, got  3584,  2560,    64,     1
```


---

## Issue #N/A: Bug: llama-server + LLava 1.6 hallucinates

**Link**: https://github.com/ggml-org/llama.cpp/issues/8001
**State**: closed
**Created**: 2024-06-19T05:40:15+00:00
**Closed**: 2024-08-03T01:18:10+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

When using `./llama-llava-cli `, I get perfectly fine descriptions of images. But when hosting LLava with `./llama-server`, LLava hallucinates big time. 

Here's how I'm running LLava with the cli:
`./llama-llava-cli -m models/llava-v1.6-vicuna-7b.Q5_K_S.gguf --mmproj models/mmproj-model-f16.gguf --image images/sth.jpeg -c 4096`

Here's how I'm starting the server:
` ./llama-server -m models/llava-v1.6-vicuna-7b.Q5_K_S.gguf --mmproj models/mmproj-model-f16.gguf -c 2048  --host 127.0.0.1 --port 8000`

Here's the python code to send the request:
```
import requests
import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("./images/sth.png")
      
headers = {
    'Content-Type': 'application/json',
}

json_data = {
    'image_data': [{
        'data': base64_image, 
        'id': 10
    }],
    "prompt": "USER:[img-

[... truncated for brevity ...]

---

## Issue #N/A: Bug: llama-perplexity error using multiple-choice binary data

**Link**: https://github.com/ggml-org/llama.cpp/issues/9316
**State**: open
**Created**: 2024-09-04T19:41:26+00:00
**Comments**: 0
**Labels**: bug, medium severity

### Description

### What happened?

"The multiple choice evaluation has been broken in llama.cpp via commit 6ff13987a.

The multiple choice evaluation uses binary data stored in params.prompt. Commit 6ff13987a adds prompt escape character processing, which modifies the binary data and renders it unusable. To preserve whatever utility 6ff13987a might have added, we add a flag indicating if the data stored in params.prompt is binary and, if so, avoid the escape processing."  @ikawrakow

@ikawrakow solved the problem in his llama.cpp fork in the following PR: https://github.com/ikawrakow/ik_llama.cpp/pull/33



### Name and Version

I tested the issue with the docker release of llama.cpp:

 ghcr.io/ggerganov/llama.cpp:full-cuda--b1-98a532d

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: Bug: ggml/src/ggml.c: In function 'ggml_vec_mad_f16':

**Link**: https://github.com/ggml-org/llama.cpp/issues/8378
**State**: closed
**Created**: 2024-07-08T21:20:30+00:00
**Closed**: 2024-08-26T01:07:03+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?
`GGML_CUDA=1 make -j`
...
```
ggml/src/ggml.c: In function 'ggml_vec_mad_f16':
ggml/src/ggml.c:2039:45: warning: passing argument 1 of '__sse_f16x4_load' discards 'const' qualifier from pointer target type [-Wdiscarded-qualifiers]
 2039 |             ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
      |                                             ^
ggml/src/ggml.c:1491:50: note: in definition of macro 'GGML_F32Cx4_LOAD'
 1491 | #define GGML_F32Cx4_LOAD(x)     __sse_f16x4_load(x)
      |                                                  ^
ggml/src/ggml.c:2039:21: note: in expansion of macro 'GGML_F16_VEC_LOAD'
 2039 |             ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
      |                     ^~~~~~~~~~~~~~~~~
ggml/src/ggml.c:1466:52: note: expected 'ggml_fp16_t *' {aka 'short unsigned int *'} but argument is of type 'const ggml_fp16_t *' {aka 'const short unsigned int *'}
 1466 | static inline __m128 __sse_f16x4_load(ggml_fp16_

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Phi-3 4K output broken after 2000~ tokens (Reproducible)

**Link**: https://github.com/ggml-org/llama.cpp/issues/7709
**State**: closed
**Created**: 2024-06-03T07:25:37+00:00
**Closed**: 2024-12-27T12:33:27+00:00
**Comments**: 13
**Labels**: bug, model, medium severity

### Description

### What happened?

To reproduce:
Download the official released gguf model from huggingface/microsoft.
Run **server.exe -m Phi3-mini-4k.gguf -c 4096**

When input prompt < ~2048: Output fine. (but output starts getting weird right after it hits ~2048 in total)
When input prompt > ~2048: Output weird.

The weird output seems like what we expect to see when the context is more than the model support, but happens in ~2048, which seems like there are some bugs.

Also tested Llama3-8B, works fine with input prompt < 8192 as expected (with -c 8192), also works fine with input prompt < 4096 as expected (with -c 4096).

### Name and Version

version: 3015 (74b239b3)
built with MSVC 19.39.33523.0 for x64

Tried both cuda and avx2 version.

Also tried latest version built it myself @ Intel SYCL
version: 3075 (3d7ebf63)
built with IntelLLVM 2024.1.0

### What operating system are you seeing the problem on?

Win10, Win11

### Relevant log output

Before ~2000 tokens 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Gemma 2 incoherent output when using quantized k cache without Flash Attention

**Link**: https://github.com/ggml-org/llama.cpp/issues/8853
**State**: closed
**Created**: 2024-08-04T10:57:51+00:00
**Closed**: 2024-09-18T01:07:04+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

Output like "Mh giàu され rodas reliablyacheteurδε Są" happens when using quantized K cache, CUDA, with Gemma 2. Here's how to reproduce:

./llama-server -m "Gemma-2-9B-It-SPPO-Iter3-Q4_K_S.gguf" -t 6 -c 8192 -ngl 31 -ctk q4_0 --host 127.0.0.1 --port 8080

Then connect a frontend like SillyTavern to it. Strangely this only happens with server, not with main-cli. 

This leads to incoherent output. Note: I can't say if this issue happens when using full offloading, as I just have 6 GB VRAM. 


### Name and Version

 ./llama-cli --version
version: 3506 (76614f35)
built with MSVC 19.29.30154.0 for x64

### What operating system are you seeing the problem on?

Windows

### Relevant log output

_No response_

---

## Issue #N/A: Bug: Last 2 Chunks In Streaming Mode Come Together In Firefox

**Link**: https://github.com/ggml-org/llama.cpp/issues/9502
**State**: closed
**Created**: 2024-09-16T02:14:04+00:00
**Closed**: 2024-09-17T06:48:47+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

When using `/completion` with `stream: true`, the last 2 JSON chunks come together in Firefox, but Chrome seems to handle it fine, so it might be a Firefox bug.

Looking further into this, it seems like HTTP `Transfer-Encoding: chunked` requires each chunk to be terminated with `\r\n`, but here `\n\n` is used instead:

https://github.com/ggerganov/llama.cpp/blob/6262d13e0b2da91f230129a93a996609a2f5a2f2/examples/server/utils.hpp#L296-L299

This doesn't seem to be just a Windows requirement, but listed as part of the HTTP specification:
[HTTP Chunked Transfer Coding](https://httpwg.org/specs/rfc9112.html#chunked.encoding)

More information, including an example `chunked` response:
[Transfer-Encoding Directives](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Transfer-Encoding#directives)

### Name and Version

llama-server.exe
version: 3761 (6262d13e)
built with MSVC 19.29.30154.0 for x64

### What operating system are you seeing the problem on?


[... truncated for brevity ...]

---

## Issue #N/A: Bug: tab/space mistokenization for gemma spm models

**Link**: https://github.com/ggml-org/llama.cpp/issues/8338
**State**: closed
**Created**: 2024-07-06T16:22:58+00:00
**Closed**: 2024-07-07T00:05:35+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

```sh
$ ./tokenize codegemma-2b.gguf "                                         test"
[snip]
     2 -> '<bos>'
255970 -> '			'
255970 -> '			'
  2121 -> ' test'
$ echo "                                             test" | spm_encode --model codegemma-2b.model --input /dev/stdin --output_format id
255973 2195
$ echo "255970 255970 2121" | spm_decode --model codegemma-2b.model --input /dev/stdin --input_format id | jq -R .
"\t\t\t\t\t\t test"
$ echo "255973 2195" | spm_decode --model codegemma-2b.model --input /dev/stdin --input_format id | jq -R .
"\t\t\t\t\t\ttest"
```

Note that the input is six tabs followed by "test", i.e. `"\t\t\t\t\t\ttest"`. Take care not to accidentally use spaces when reproducing.

Note that this is not _just_ inserting a stray space before "test": it also breaks the tabs into two sets of 3 instead of a single set of 6.

Inputs like this (leading indentation followed by text) happen a lot with code.

There are three iss

[... truncated for brevity ...]

---

## Issue #N/A: Bug: ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 137438953504

**Link**: https://github.com/ggml-org/llama.cpp/issues/8101
**State**: closed
**Created**: 2024-06-24T20:30:57+00:00
**Closed**: 2024-06-24T21:36:37+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

I'm working on a project with llama.cpp to process a bunch of text files and I'm trying to use multi processing to speed up, so I'm loading the model and context in a child process for each file, launching 10 children at a time but some times one the child process would fail to create the context with the following error:

`llama_new_context_with_model: n_ctx      = 1048576
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 2804339712.0
llama_new_context_with_model: freq_scale = 1
ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 137438953504
llama_kv_cache_init: failed to allocate buffer for kv cache
llama_new_context_with_model: llama_kv_cache_init() failed for self-attention cache`

Isn't 137438953504 bytes way too much memory to try to allocate. If I run the files one by one without using child

[... truncated for brevity ...]

---

## Issue #N/A: Bug: symbols conflict with whisper.cpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/9267
**State**: closed
**Created**: 2024-09-01T17:07:16+00:00
**Closed**: 2024-10-29T01:07:29+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

I'm trying to use llama.cpp alongside whisper.cpp in Rust but I can't link the libraries because they link the same ggml symbols.

[llama-cpp-2](https://github.com/utilityai/llama-cpp-rs/blob/main/llama-cpp-sys-2/build.rs) and [whisper-rs](https://github.com/tazz4843/whisper-rs)



### Name and Version

8f1d81a0b6f50b9bad72db0b6fcd299ad9ecd48c
https://github.com/ggerganov/whisper.cpp/commit/c96906d84dd6a1c40ea797ad542df3a0c47307a3

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
b\\rustlib\\etc\\libstd.natvis"
  = note: libllama_cpp_sys_2-1131e8f6e28c8598.rlib(ggml-backend.obj) : error LNK2005: ggml_backend_buft_name already defined in libwhisper_rs_sys-f2fa5877d4809bf3.rlib(ggml-backend.obj)
          libllama_cpp_sys_2-1131e8f6e28c8598.rlib(ggml-backend.obj) : error LNK2005: ggml_backend_buft_alloc_buffer already defined in libwhisper_rs_sys-f2fa5877d4809bf3.rlib(ggml-backend.obj)
```

I

[... truncated for brevity ...]

---

## Issue #N/A: Bug -  Can't build vulkan backend on RISC-V platform anymore

**Link**: https://github.com/ggml-org/llama.cpp/issues/8488
**State**: closed
**Created**: 2024-07-15T07:08:16+00:00
**Closed**: 2024-09-22T01:07:39+00:00
**Comments**: 8
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

Bug: Can't build vulkan backend on RISC-V platform anymore

### Name and Version

git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
apt-get install cmake
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release -j8

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
Vulkan_GLSLC_EXECUTABLE-NOTFOUND -fshader-stage=compute --target-env=vulkan1.2 -O /root/liyong/llama.cpp/ggml/src/vulkan-shaders/mul_mm.comp -o /root/liyong/llama.cpp/build/ggml/src/vulkan-shaders.spv/matmul_f32_f16_aligned_fp32.spv -DB_TYPE=f16vec4 -DDATA_A_F32=1 -DD_TYPE=float -DFLOAT_TYPE=float -DLOAD_VEC_A=4 -DLOAD_VEC_B=4

sh: 1: Vulkan_GLSLC_EXECUTABLE-NOTFOUND: not found

cannot compile matmul_f32_f32_aligned_fp32

Vulkan_GLSLC_EXECUTABLE-NOTFOUND -fshader-stage=compute --target-env=vulkan1.2 -O /root/liyong/llama.cpp/ggml/src/vulkan-shaders/mul_mm.comp -o /root/liyong/llama.cpp/build/ggml/src/vulkan-shade

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Processor features are determined at compile time

**Link**: https://github.com/ggml-org/llama.cpp/issues/9147
**State**: closed
**Created**: 2024-08-23T10:14:13+00:00
**Closed**: 2024-10-11T01:07:16+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

I'm running ollama which in turn uses llama.cpp. The server has quad Intel Xeon Sapphire rapids. In the debug line for the "system info" i get:

```shell
INFO [main] system info | n_threads=160 n_threads_batch=-1 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 0 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | " tid="139749882310656" timestamp=1724406025 total_threads=320
```

Which wondered me as the SPR processors have (from /proc/cpuinfo):

```shell
fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: gemma2 perplexity pending forever

**Link**: https://github.com/ggml-org/llama.cpp/issues/8490
**State**: closed
**Created**: 2024-07-15T11:14:05+00:00
**Closed**: 2024-08-29T01:07:02+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

I am attempting to measure the perplexity of the gemma-2-9b-it-Q4_K_M.gguf model using llama.cpp. However, I encounter an issue where the process gets stuck at the "tokenizing the input" stage indefinitely.

I have confirmed that the qwen2-7b-instruct-q4_k_m.gguf model operates correctly in the same environment, so I expected gemma-2 to function properly as well. Unfortunately, it does not.

the model is from huggingface model hub,
 bartowski/gemma-2-9b-it-GGUF

### more information
I just found out that the data I have is a Korean Wikipedia dataset, and it worked fine with qwen2, but it doesn't seem to work with gemma2. After changing the data to a wiki.test.raw file, I confirmed that it works properly

I also discovered that the original number of files was 10,000, but after reducing it to 500, it worked. It seems to operate much slower compared to Qwen.


지미 카터
Introduction


'''제임스 얼 “지미” 카터 주니어'''(, 1924년 10월 1일~)는 민주당 출신 미국의 제39대 대통령 (1977-8

[... truncated for brevity ...]

---

## Issue #N/A: Bug: ggml_vulkan can only Found 1 Vulkan devices.

**Link**: https://github.com/ggml-org/llama.cpp/issues/9716
**State**: closed
**Created**: 2024-10-02T14:38:24+00:00
**Closed**: 2024-11-18T01:07:45+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

I have two Vulkan devices  NVIDIA GeForce RTX 3060 Laptop GPU (NVIDIA) and AMD Radeon(TM) Graphics (AMD proprietary driver),but **ggml_vulkan can only found one.**
In general,the cli output is :
```shell
ggml_vulkan: Found 1 Vulkan devices:
Vulkan0: NVIDIA GeForce RTX 3060 Laptop GPU (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32
llm_load_tensors: ggml ctx size =    0.19 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/37 layers to GPU
llm_load_tensors:        CPU buffer size =  3442.89 MiB
``` 
If I disable the NVIDIA in system device manager,then start llamacpp again,ggml_vulkan can found another device:
```shell
ggml_vulkan: Found 1 Vulkan devices:
Vulkan0: AMD Radeon(TM) Graphics (AMD proprietary driver) | uma: 1 | fp16: 1 | warp size: 64
llm_load_tensors: ggml ctx size =    0.19 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/37 layers to GPU
llm_load_tensors:        

[... truncated for brevity ...]

---

## Issue #N/A: Bug: tokenizer.chat_template missing from key/values

**Link**: https://github.com/ggml-org/llama.cpp/issues/8403
**State**: closed
**Created**: 2024-07-09T23:09:25+00:00
**Closed**: 2024-07-10T12:59:30+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

I am seeing an issue with the default chat templates not properly loading. It appears as though the `tokenizer.chat_template` key is missing from the model meta-data. Here is a test program showcasing the issue:

```cpp
    const char * model = "D:\\models\\llama-2-7b-chat.Q5_K_M.gguf";
    struct llama_model_params params = llama_model_default_params();
    params.use_mmap = false;

    struct llama_model * model_ = llama_load_model_from_file(model, params);
    if (! model_) {
        printf("uh oh");
        exit(1);
    }

    // This snippet is from recent pull request.
    std::string template_key = "tokenizer.chat_template", curr_tmpl;
    int32_t tlen2 = llama_model_meta_val_str(model_, template_key.c_str(), nullptr, 0);
    if (tlen2 > 0) {
        std::vector<char> curr_tmpl_buf(tlen2 + 1, 0);
        if (llama_model_meta_val_str(model_, template_key.c_str(), curr_tmpl_buf.data(), curr_tmpl_buf.size()) =\
= tlen2) {
            curr_t

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Unable to load model using SYCL

**Link**: https://github.com/ggml-org/llama.cpp/issues/7968
**State**: closed
**Created**: 2024-06-17T03:11:56+00:00
**Closed**: 2024-06-18T06:35:40+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

I am using the commit `0c7b359`
Build llama.cpp using the command below:
```
source /opt/intel/oneapi/setvars.sh
cmake -B build -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release -j -v
```
Trying to run a llama-2-7b-model.gguf model but got `Segmentation fault (core dumped)`
Command:
```
./build/bin/llama-cli --model llama-2-7b-model.gguf --gpu-layers 33
```

### Name and Version

version: 3153 (0c7b3595)
built with Intel(R) oneAPI DPC++/C++ Compiler 2024.0.2 (2024.0.2.20231213) for x86_64-unknown-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
main: build = 3153 (0c7b3595)
main: built with Intel(R) oneAPI DPC++/C++ Compiler 2024.0.2 (2024.0.2.20231213) for x86_64-unknown-linux-gnu
main: seed  = 1718593892
llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from llama-2-7b-model.gguf (version GGUF V2)

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

## Issue #N/A: Bug: llama-cli does not show the results of the performance test when SIGINT

**Link**: https://github.com/ggml-org/llama.cpp/issues/9558
**State**: closed
**Created**: 2024-09-20T03:28:12+00:00
**Closed**: 2024-09-20T08:46:57+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

When running llama-cli in conversation mode, press Ctrl+C to interject it did not print the results of the performance info.

git bisect show it's 6262d13e0b2da91f230129a93a996609a2f5a2f2.

e6deac31f7e62db43b6afbc3be814f764fd5a187
```
>
llama_perf_sampler_print:    sampling time =      17.31 ms /   124 runs   (    0.14 ms per token,  7164.32 tokens per second)
llama_perf_context_print:        load time =    2548.25 ms
llama_perf_context_print: prompt eval time =    4104.68 ms /    25 tokens (  164.19 ms per token,     6.09 tokens per second)
llama_perf_context_print:        eval time =    6035.09 ms /   109 runs   (   55.37 ms per token,    18.06 tokens per second)
llama_perf_context_print:       total time =   36065.20 ms /   134 tokens
localhost:~/code/kleidiai/llama.cpp #
```

6262d13e0b2da91f230129a93a996609a2f5a2f2: (empty output after Ctrl+C)
```
> localhost:~/code/kleidiai/llama.cpp #
```



### Name and Version

localhost:~/code/kleidiai/

[... truncated for brevity ...]

---

## Issue #N/A: Bug: null pointer defer in gguf_init_from_file

**Link**: https://github.com/ggml-org/llama.cpp/issues/8583
**State**: closed
**Created**: 2024-07-19T04:49:43+00:00
**Closed**: 2024-07-20T14:15:43+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

``` cpp
            ok = ok && gguf_fread_str(file, &info->name,                          &offset);   // [1] maybe read failed,then info->name = nullptr
            ok = ok && gguf_fread_el (file, &info->n_dims, sizeof(info->n_dims),  &offset);

            ok = ok && (info->n_dims <= GGML_MAX_DIMS);

            for (uint32_t j = 0; j < info->n_dims; ++j) {
                ok = ok && gguf_fread_el(file, &info->ne[j], sizeof(info->ne[j]), &offset);
            }

            ok = ok && gguf_fread_el (file, &info->type,   sizeof(info->type),    &offset);
            ok = ok && gguf_fread_el (file, &info->offset, sizeof(info->offset),  &offset);

            // TODO: return an error instead of crashing with GGML_ASSERT
            gguf_tensor_info_sanitize(info);

            // make sure there is no duplicated tensor names
            for (uint64_t j = 0; j < i; ++j) {
                if (strcmp(info->name.data, ctx->infos[j].name.data) == 0) {     

[... truncated for brevity ...]

---

## Issue #N/A: Encountering some errors while using Android NDK with Vulkan

**Link**: https://github.com/ggml-org/llama.cpp/issues/7760
**State**: closed
**Created**: 2024-06-05T05:51:03+00:00
**Closed**: 2024-07-20T01:06:42+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

I am cross-compiling llama.cpp using android-ndk-r25c and Vulkan, but encountering the following error during compilation.

I found that there is a Vulkan library in NDK-r25c. I included the header file from it. Additionally, I have installed Vulkan SDK on my computer.

![image](https://github.com/ggerganov/llama.cpp/assets/46549527/7840a78b-f773-4a81-b56f-0ec21200efcb)


![image](https://github.com/ggerganov/llama.cpp/assets/46549527/c4e0d084-b75e-401e-961d-2bd7ddd87607)


/home/zdl/AIGC_LLAMA_Project/llama.cpp/ggml-vulkan.cpp:164:16: error: call to member function 'destroyCommandPool' is ambiguous
        device.destroyCommandPool(compute_queue.pool);
        ~~~~~~~^~~~~~~~~~~~~~~~~~
/home/zdl/AndroidSDK/android-ndk-r25c/sources/third_party/vulkan/src/include/vulkan/vulkan.hpp:84770:34: note: candidate function [with Dispatch = vk::DispatchLoaderStatic]
  VULKAN_HPP_INLINE void Device::destroyCommandPool( VULKAN_HPP_NAMESPACE::CommandPool commandPoo

[... truncated for brevity ...]

---

## Issue #N/A: Bug: when arrive max ctx, model output garbage

**Link**: https://github.com/ggml-org/llama.cpp/issues/7578
**State**: closed
**Created**: 2024-05-28T02:22:16+00:00
**Closed**: 2024-06-18T03:20:17+00:00
**Comments**: 2
**Labels**: need more info, bug-unconfirmed, medium severity

### Description

### What happened?

This part has problem in cuda version. if set ngl>0, when arrive max ctx and next turn to chat, the model output garbage.

llama_kv_cache_seq_rm (ctx, 0, params.n_keep            , params.n_keep + n_discard);
llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

if set ngl =0, everythings ok.
### Name and Version

llama.cpp-b3014
main.exe --version
version: 247 (6765407)
built with MSVC 19.37.32822.0 for x64

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Gemma 2 inference - continuous token succession

**Link**: https://github.com/ggml-org/llama.cpp/issues/8324
**State**: closed
**Created**: 2024-07-05T09:51:17+00:00
**Closed**: 2024-07-05T14:25:27+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

There is an issue in which after the models answers the question it keeps on generating 2 tokens until the context is filled up.
Tokens in question:
    {
      "id": 149,
      "content": "▁▁▁▁▁▁▁▁▁▁▁▁",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 108,
      "content": "\n",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    
   How the server is run:
   llama-server  --threads 1 --batch-size 256 --threads-batch 32 --n-gpu-layers 43 --main-gpu 0 --metrics --defrag-thold 0.8 --cont-batching -m  /<model_path>/gemma-2-9b-it/gemma-2-9b-it-Q6_K.gguf -v -c 32786 -n 1024  --parallel 16   --host <h>  --port <p>

The query to the model is made via the /completion API

The model is used in a RAG system, the following prompt has been used:
        prompt = '''
  

[... truncated for brevity ...]

---

## Issue #N/A: Bug: [SYCL] Error loading models larger than Q4

**Link**: https://github.com/ggml-org/llama.cpp/issues/9472
**State**: closed
**Created**: 2024-09-13T13:48:33+00:00
**Closed**: 2024-10-30T01:19:50+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

After building the SYCL server image, trying to load a model larger than Q4 on my Arc A770 fails with a memory error.
Anything below Q4 will execute, but this is due to the "llm_load_tensors:      SYCL0 buffer size" being below ~4200MiB.
The Arc A770 has 16GB of VRAM, so should be perfectly capable of loading much higher buffer values into its VRAM.

Looking for information on this. Thanks!

### Name and Version

Relevant docker run command used:
docker run -it --rm -p 11434:11434 -v /mnt/user/models/model-files:/app/models --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card0:/dev/dri/card0 -e OverrideGpuAddressSpace=48 -e NEOReadDebugKeys=1 llama-server-cpp-intel -m /app/models/Meta-Llama-3.1-8B-Instruct-Q5_K_L.gguf -n 2048 -e -ngl 33 --port 11434

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
llama_model_loader: - kv  29:                      quantize.imatrix.file str            

[... truncated for brevity ...]

---

## Issue #N/A: Bug: When inferring with RWKV, an uncontrolled dialogue between 'user' and 'assistant' appears.

**Link**: https://github.com/ggml-org/llama.cpp/issues/9892
**State**: closed
**Created**: 2024-10-15T02:17:30+00:00
**Closed**: 2024-10-22T10:33:38+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

When I execute this command **./llama-cli -m rwkv6-3b.gguf -p hello -cnv** and say“您好，你是谁？”, it answers me:“你好,我是助手。”，Then everything **below is generated by itself**, without any action from me. and it uncontrollably generates a dialogue between 'user' and 'Assistant', **as shown below:**

<|im_start|>system
hello<|im_end|>

> 您好,你是谁?
你好,我是助手。
<|im_end|>
<|im_start|>hello
您好。
<|im_end|>
<|im_start|>world
你好,世界。

Assistant: 非常感谢您使用我的服务，如果您有其他问题需要帮助，随时欢迎向我提问。祝您一切顺利！User: 写一篇有关环保的文章。

Assistant: 好的，以下是一篇有关环保的文章：
在当今社会，环保已经成为我们生活中不可忽视的问题。随着人口的不断增长和工业的不断发展，我们的生活方式和工作方式已经对环境造成了极大的负面影响。因此，我们需要采取行动来保护我们的环境和地球。
首先，我们需要意识到我们的行为对环境的影响。我们需要减少垃圾和废水的产生，避免浪费食物和能源。我们可以通过使用环保袋和水瓶来减少塑料垃圾和能源消耗。此外，我们也可以选择使用环保型产品和材料，例如使用可降解的塑料和环保型电器。
其次，我们需要采取行动来保护我们的环境。我们可以参加各种环保活动和志愿者活动，例如清理公园和河流、种树和种草等。我们也可以通过投票、支持政府的环保政策和倡导环保的行为来推动环保事业的发展。
最后，我们需要教育我们的下一代。我们需要让他们明白环保的重要性，并教他们如何保护我们的环境。我们可以在学校和社区组织环保讲座、研讨会和活动，以便他们了解环保的重要性和如何为环保做出贡献。
在未来，我们需要采取更加积极和有效的措施来保护我们的环境和地球。只有这样，我们才能保护我们的

[... truncated for brevity ...]

---

## Issue #N/A: Bug: rwkv and mamba models cannot be used with `-ngl 0` after CPU backend refactor

**Link**: https://github.com/ggml-org/llama.cpp/issues/10351
**State**: closed
**Created**: 2024-11-17T02:47:58+00:00
**Closed**: 2024-11-17T11:25:46+00:00
**Comments**: 1
**Labels**: bug, medium severity

### Description

### What happened?

```
$ ./build/bin/llama-bench -m ~/Downloads/mamba-2.8b-q4_0.gguf -ngl 0
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
/Users/molly/llama.cpp/ggml/src/ggml-backend.cpp:745: pre-allocated tensor in a backend that cannot run the operation
[1]    13345 abort      ./build/bin/llama-bench -m ~/Downloads/mamba-2.8b-q4_0.gguf -ngl 0
```
```
$ ./build/bin/llama-bench -m /Volumes/grouped/Models/rwkv/v6-Finch-7B-HF/v6-Finch-7B-HF-Q4_0.gguf -ngl 0
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
/Users/molly/llama.cpp/ggml/src/ggml-backend.cpp:745: pre-allocated tensor in a ba

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Gemma2 adapter weights `lm_head` skipped on gguf conversion

**Link**: https://github.com/ggml-org/llama.cpp/issues/9065
**State**: closed
**Created**: 2024-08-17T13:10:30+00:00
**Closed**: 2024-09-12T11:33:58+00:00
**Comments**: 10
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

The `lm_head` layer for a [Gemma2](https://huggingface.co/google/gemma-2-2b) LoRA adapter is not converted by `convert_lora_to_gguf.py`, and therefore not applied at inference (ruining performance of the adapter).

<br>

### How to reproduce:


<details>
<summary>Expand</summary>

<br>

1. LoRA fine-tune Gemma2 with `pytorch`/`peft` including `lm_head` in the `target_modules` param:
    ```python
    config = LoraConfig(target_modules=["lm_head"], ...)
    ``` 
2. Save the adapter.
3. Convert the adapter debugging  
    ```bash
    python convert_lora_to_gguf.py <adapter folder> --base <base model folder> --outtype f32
    ```
    then the `lm_head` layer is skipped by [this line in `convert_hf_to_gguf.py`](https://github.com/ggerganov/llama.cpp/blob/4b9afbbe9037f8a2d659097c0c7d9fce32c6494c/convert_hf_to_gguf.py#L2648) (and no error is raised):
    ```python
    if name == "lm_head.weight":
       logger.debug(f"Skipping get tensor {name!r}

[... truncated for brevity ...]

---

