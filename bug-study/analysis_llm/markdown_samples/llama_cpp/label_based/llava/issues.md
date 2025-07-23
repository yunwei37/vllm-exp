# llava - issues

**Total Issues**: 6
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 5

### Label Distribution

- llava: 6 issues
- enhancement: 3 issues
- stale: 3 issues
- server/webui: 2 issues
- bug-unconfirmed: 2 issues
- server: 1 issues
- help wanted: 1 issues
- good first issue: 1 issues

---

## Issue #N/A: server: Bring back multimodal support

**Link**: https://github.com/ggml-org/llama.cpp/issues/8010
**State**: closed
**Created**: 2024-06-19T12:03:45+00:00
**Closed**: 2025-05-09T21:20:01+00:00
**Comments**: 51
**Labels**: enhancement, llava, server

### Description

Multimodal has been removed since https://github.com/ggerganov/llama.cpp/pull/5882

## Current llama.cpp multimodal roadmap

(update 9th april 2025)

- `mtmd` (**M**ul**T**i-**M**o**D**al) library (top prio 🔥 )
    - [x] Implement `libmtmd`: https://github.com/ggml-org/llama.cpp/pull/12849
    - [x] Support more models via `libmtmd` (top prio 🔥 ) : https://github.com/ggml-org/llama.cpp/pull/13012
    - [x] Support M-RoPE models via `libmtmd` (Qwen2VL, Qwen2.5VL) : https://github.com/ggml-org/llama.cpp/pull/13141
    - [x] Support audio input
    - [x] Use smart pointer in `clip.cpp` to avoid mem leak: https://github.com/ggml-org/llama.cpp/pull/12869
    - [x] ~~Add wrapper for `stb_image` to avoid polluting project with the big header file~~ --> Probably don't need since we're already having some helper in `libmtmd` acting as wrapper for stb_image
    - [x] Unify conversion scripts --> best case scenario: having `convert_hf_to_gguf.py` that can output both text + vision GGUF files --> 

[... truncated for brevity ...]

---

## Issue #N/A: Add multimodal example

**Link**: https://github.com/ggml-org/llama.cpp/issues/6313
**State**: closed
**Created**: 2024-03-26T02:50:13+00:00
**Closed**: 2024-03-26T06:32:36+00:00
**Comments**: 2
**Labels**: enhancement, llava

### Description

# Feature Description

Add example for multimodal capabilities

# Motivation

#5882 took out the multimodal features from the server. Given it's a highly requested feature, our plan would be to reintroduce it at some point (#6168). How about we set up a solid multimodal example elsewhere and then port it to the server example later on?

# Possible Implementation

Implementation based on the removed code from https://github.com/ggerganov/llama.cpp/pull/5882/files which had already implemented this feature in the server.cpp example, hopefully with some performance optimization.
For the example, image file could be provided via command line option.


---

## Issue #N/A: Unable to assign mmproj value when running docker 

**Link**: https://github.com/ggml-org/llama.cpp/issues/6226
**State**: closed
**Created**: 2024-03-22T07:52:31+00:00
**Closed**: 2024-05-07T01:06:30+00:00
**Comments**: 2
**Labels**: server/webui, bug-unconfirmed, stale, llava

### Description

Please include information about your system, the steps to reproduce the bug, and the version of llama.cpp that you are using. If possible, please provide a minimal code example that reproduces the bug.

If the bug concerns the server, please try to reproduce it first using the [server test scenario framework](https://github.com/ggerganov/llama.cpp/tree/master/examples/server/tests).

Command
```sh
sudo docker run -p 5000:8000  --gpus all --runtime=nvidia -v /models:/models ghcr.io/ggerganov/llama.cpp:server-cuda -m /models/ggml-model-q4_k.gguf --mmproj /models/mmproj-model-f16.gguf  --port 8000 --host 0.0.0.0 -v  -t 16  -n 512 -c 2048 -ngl 1 -cb -np 4 --n-gpu-layers 33
```

Error
```sh
error: unknown argument: --mmproj
```

--mmproj option is not supported by docker. 

The documentation mentions this option though.
https://github.com/ggerganov/llama.cpp/tree/master/examples/server#llamacpp-http-server


---

## Issue #N/A: llava-cli: improve llava-cli and the API for using LLaVA

**Link**: https://github.com/ggml-org/llama.cpp/issues/6027
**State**: open
**Created**: 2024-03-12T21:18:55+00:00
**Comments**: 4
**Labels**: enhancement, help wanted, good first issue, llava

### Description

From:
 - https://github.com/ggerganov/llama.cpp/issues/4216#issuecomment-1991730224

1. cleaning up the clip/llava libs and improving the API
2. in the old implementation, there were many internal object exposed to the server and the memory management was dubious
3. there was no obvious path for supporting parallel multimodal slots


---

## Issue #N/A: llama cpp server not doing parallel inference for llava when using flags -np and -cb

**Link**: https://github.com/ggml-org/llama.cpp/issues/5592
**State**: closed
**Created**: 2024-02-19T18:16:43+00:00
**Closed**: 2024-05-07T01:06:42+00:00
**Comments**: 11
**Labels**: server/webui, bug-unconfirmed, stale, llava

### Description

When I am trying to do parallel inferencing on llama cpp server for multimodal, I am getting the correct output for slot 0, but for other slots, I am not. Does that mean that clip is only being loaded on one slot? I can see some clip layers failing to load.

Here is my llama cpp server code that I use.

`./server -m ../models/llava13b1_5/llava13b1_5_f16.gguf -c 40960 --n-gpu-layers 41 --port 8001 --mmproj ../models/llava13b1_5/llava13b1_5_mmproj_f16.gguf -np 10 -cb --host 0.0.0.0 --threads 24`

The model I am using - 
[https://huggingface.co/mys/ggml_llava-v1.5-13b/tree/main](model)

I am using the F16 model with mmproj file.

Documentation reference

[https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md](documentation)

My GPU specs

![image](https://github.com/ggerganov/llama.cpp/assets/137015071/c7e6506e-1261-47a5-85c3-665d75fe3e7d)

My CPU specs

![image](https://github.com/ggerganov/llama.cpp/assets/137015071/8169172c-6ac3-4bea-a2f7-626

[... truncated for brevity ...]

---

## Issue #N/A: Running Lllava in interactive mode just Quits after generating response without waiting for next prompt.

**Link**: https://github.com/ggml-org/llama.cpp/issues/3593
**State**: closed
**Created**: 2023-10-12T04:23:58+00:00
**Closed**: 2024-10-02T01:11:36+00:00
**Comments**: 15
**Labels**: stale, llava

### Description

What the title said.

`llava -m ../models/llava-13b-q5_k.gguf --mmproj ../models/mmproj-model-f16.gguf -i -r "user:" --rope_freq_base 0 --rope_freq_scale 0 --temp 0.1 --top-p 0.9 --top-k 90 --image test.png -p "Describe the image in detail."`

It describes the image and quits. Then I get a thrown into a command line prompt.

How can I run in interactive mode, so I can ask more about the response?

Thanks so much!

---

