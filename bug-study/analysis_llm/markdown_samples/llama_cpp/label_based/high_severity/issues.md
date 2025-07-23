# high_severity - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- high severity: 30 issues
- bug-unconfirmed: 28 issues
- stale: 14 issues
- documentation: 1 issues
- script: 1 issues
- python: 1 issues
- bug: 1 issues

---

## Issue #N/A: Bug: Decoding special tokens in T5

**Link**: https://github.com/ggml-org/llama.cpp/issues/8938
**State**: closed
**Created**: 2024-08-08T16:32:39+00:00
**Closed**: 2024-08-09T16:53:10+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

I have a T5/lora model trained to output some text separated by the `<extra_id_0>` special token (the tokenizer properly works after following instructions in #8872) .

When running the model using Huggingface's transformers/peft, it generates the expected output. However, when I use `llama-cli`, what happens instead is that the moment the first such token is reached, it's actually decoded into an `EOG` token instead of the extra token and generation is stopped.

I might be simply doing something wrong in using the library.

### Name and Version

version: 3549 (afd27f01)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu


### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Bug: not support langchain v0.3 to use tools

**Link**: https://github.com/ggml-org/llama.cpp/issues/10214
**State**: closed
**Created**: 2024-11-08T08:59:18+00:00
**Closed**: 2024-12-23T01:30:30+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

request: request: POST /v1/chat/completions 192.168.139.86 500

request:
{
	"messages": [{
		"content": "98平米的房屋总价是多少",
		"role": "user"
	}],
	"model": "qwen-plus",
	"n": 1,
	"stream": false,
	"temperature": 0.7,
	"tools": [{
		"type": "function",
		"function": {
			"name": "magic_function",
			"description": "根据房屋面积，计算房屋价格。input 是房屋面积单位是平米，返回的结果是房屋价格，单位是元",
			"parameters": {
				"properties": {
					"input": {
						"type": "integer"
					}
				},
				"required": ["input"],
				"type": "object"
			}
		}
	}]
}

response:

{
	"error": {
		"code": 500,
		"message": "Unsupported param: tools",
		"type": "server_error"
	}
}


### Name and Version

(base) [root@localhost llama.cpp-master]# ./llama-cli --version
version: 0 (unknown)
built with cc (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2) for x86_64-redhat-linux

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
request: POST /v1/chat/c

[... truncated for brevity ...]

---

## Issue #N/A: Bug: phi-3-mini-4k-it July update failing to load.

**Link**: https://github.com/ggml-org/llama.cpp/issues/8845
**State**: closed
**Created**: 2024-08-03T12:32:44+00:00
**Closed**: 2024-08-05T11:35:13+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

i am trying to load the phi-3-mini july update model as usual but its giving me the following error:

```
llama_model_load: error loading model: error loading model hyperparameters: key not found in model: phi3.attention.sliding_window
llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model '.\models\me\phi-3-mini-4k-it-July-5\Phi-3.1-mini-4k-instruct-Q8_0_L.gguf'
main: error: unable to load model
```

Also, phi-2 and phi-3 original model still work! If its worth knowing, i have also downloaded the latest version of LM Studio, and its also unable to run this same model, throwing the same error.

### Name and Version

PS F:\ai3> .\llama.cpp\build\bin\Release\llama-cli.exe --version
version: 3505 (b72c20b8)
built with MSVC 19.40.33811.0 for x64

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
PS F:\ai3> .\llama.cpp\build\bin\Release\llama-cli

[... truncated for brevity ...]

---

## Issue #N/A: Bug: <extra_id_i> tokens not handled properly in T5 models.

**Link**: https://github.com/ggml-org/llama.cpp/issues/8872
**State**: closed
**Created**: 2024-08-05T12:43:07+00:00
**Closed**: 2024-08-07T19:02:43+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

I am using a [quantized gguf model](https://huggingface.co/cyanic-selkie/flan-t5-small-Q8_0-GGUF/tree/main) generated via the [Huggingface space](https://huggingface.co/spaces/ggml-org/gguf-my-repo).

I am also using a LoRA adapter fine tuned for a specific task using the special `<extra_id_i>` tokens that you can see in the original `google/flan-t5-small` repo's [tokenizer.json](https://huggingface.co/google/flan-t5-small/raw/main/tokenizer.json).

From the `main.log` file, it is clear that these special tokens aren't properly tokenized, but are treated as regular strings.

### Name and Version

version: 3520 (d3f0c716)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu


### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
[1722861728] prompt: "<extra_id_0>World War II ended in 1945.<extra_id_1>What ended in 1945?<extra_id_2>World War II.<extra_id_3>The American Revolutionary War."
[1722861

[... truncated for brevity ...]

---

## Issue #N/A: Why is convert.py missing?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7658
**State**: closed
**Created**: 2024-05-31T05:46:36+00:00
**Closed**: 2024-06-10T19:58:23+00:00
**Comments**: 15
**Labels**: documentation, script, python, high severity

### Description

### What happened?

Critical "non llama3" convert.py and change NOT in download of files.

ALSO:
It is unclear if "convert-hf-to-gguf.py"  supports what "convert.py" did . 

Does it convert llama, llama2, mistral or is "convert-legacy-llama.py" required?
Safetensor files are EVERYWHERE. (?)  [ RE: https://github.com/ggerganov/llama.cpp/pull/7430 ]

This critical action DID NOT OCCUR:
"Move convert.py to examples/convert-legacy-llama.py"

"examples/convert-legacy-llama.py" does not exist. 
(when downloading the zip files).

On another note why remove "convert.py" at all? 

-This breaks "bat files" and automation generation.
-This will break all colabs too.
-This will break any HF spaces that create GGUF files as well.
-This will create needless confusion.

If "convert-hf-to-gguf.py" (llama3) does everything convert.py did , just keep it as "convert.py" ?





### Name and Version

this is not applicable - core files missing.

### What operating system are you se

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Cannot build with C++ > 20

**Link**: https://github.com/ggml-org/llama.cpp/issues/9944
**State**: closed
**Created**: 2024-10-18T16:36:32+00:00
**Closed**: 2024-12-26T01:07:27+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

Hi there,
I was trying to build llama.cpp in a project that uses the C++ 23 standard and there are a lot of errors when building the `llama` target with MSVC. The only fix is to downgrade the standard to a max of C++ 17. I'm not exactly sure how to solve this error, so any help is appreciated.

I've included the entire build log below and I'm happy to run more tests if necessary.

### Name and Version

llama.cpp repository afd9909a6481402844aecefa8a8908afdd7f52f1
VS 2022 community
MSVC v143

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
"C:\Program Files\JetBrains\CLion 2024.2.2\bin\cmake\win\x64\bin\cmake.exe" --build D:\llama.cpp\cmake-build-debug-visual-studio --target llama -j 10
[0/1] Re-running CMake...
-- OpenMP found
-- Using llamafile
CMake Warning at ggml/src/CMakeLists.txt:274 (message):
  AMX requires gcc version > 11.0.  Turning off GGML_AMX.


-- Warning: ccache not found - consider in

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Latest version of convert_hf_to_gguf not compatible with gguf 0.9.1 from pip

**Link**: https://github.com/ggml-org/llama.cpp/issues/8925
**State**: closed
**Created**: 2024-08-08T09:03:16+00:00
**Closed**: 2024-09-22T01:07:29+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

Hey,
I've been trying to run the convert_hf_to_gguf.py in python 3.12 in my development enviroment.
While trying to run a model conversion I got a error from  GGUFWriter.
I investigated the pip package from the official pip repo:
https://files.pythonhosted.org/packages/9f/1e/6bd024e0138d663cd333e8fde4c03d343e8be680d43b86537ef1497a7e32/gguf-0.9.1.tar.gz
The wheel has the same issue.
And it seems that the published version does not contain the latest changes from master.
Can you guys bump the version of gguf and publish it to pip?
Thanks

### Name and Version

version: 3541 (be55695)
built with gcc (GCC) 11.2.1 20220127 (Red Hat 11.2.1-9) for x86_64-redhat-linux

### What operating system are you seeing the problem on?

Linux Red Hat 11.2.1-9

### Relevant log output

```shell
Traceback (most recent call last):
  File "/app/./convert_hf_to_gguf.py", line 3823, in <module>
    main()
  File "/app/./convert_hf_to_gguf.py", line 3817, in main


[... truncated for brevity ...]

---

## Issue #N/A: Bug: ggml_vulkan: Failed to allocate pinned memory.

**Link**: https://github.com/ggml-org/llama.cpp/issues/9271
**State**: closed
**Created**: 2024-09-02T08:52:10+00:00
**Closed**: 2024-10-27T01:09:53+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

llama-cpp prints this error when larger models are imported:
```
ggml_vulkan: Failed to allocate pinned memory.
ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory
```

The complete log is:
```
$ llama-server -m llama-2-7b-chat.Q4_K_M.gguf --host 0.0.0.0 --port 9011
INFO [                    main] build info | tid="0x368162012000" timestamp=1725256043 build=0 commit="unknown"
INFO [                    main] system info | tid="0x368162012000" timestamp=1725256043 n_threads=4 n_threads_batch=4 total_threads=8 system_info="AVX = 0 | AVX_VNNI = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 0 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
INFO [                    main] HTTP server is listening | tid="0x368162012000" timestamp=1725256043 port="9011" n_threads_http="7" hostname="0.0.0.0"
INFO [    

[... truncated for brevity ...]

---

## Issue #N/A: Bug: [SYCL] linker fails with undefined reference to symbol

**Link**: https://github.com/ggml-org/llama.cpp/issues/9490
**State**: closed
**Created**: 2024-09-15T06:24:16+00:00
**Closed**: 2024-09-16T01:41:33+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

With the latest master branch, the linker fails with "undefined reference to symbol '_ZNK4sycl3_V16device8get_infoINS0_3ext5intel4info6device9device_idEEENS0_6detail19is_device_info_descIT_E11return_typeEv'"


### Name and Version

Latest Master; commit hash: 822b6322dea704110797a5671fc80ae39ee6ac97
intel-oneapi-basekit:  2024.1.0.596-3
Intel Compute Runtime: 24.31.30508.7-1

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
[100%] Linking CXX executable ../../bin/llama-server
cd /home/qnixsynapse/Public/Projects/llama.cpp/build/examples/server && /usr/bin/cmake -E cmake_link_script CMakeFiles/llama-server.dir/link.txt --verbose=1
/opt/intel/oneapi/compiler/2024.1/bin/icpx -O3 -DNDEBUG "CMakeFiles/llama-server.dir/server.cpp.o" -o ../../bin/llama-server  ../../common/libcommon.a ../../src/libllama.a ../../ggml/src/libggml.a /opt/intel/oneapi/compiler/2024.1/lib/libiomp5.so /usr/lib64/libpthread.a /

[... truncated for brevity ...]

---

## Issue #N/A: Bug: CUDA illegal memory access related to KV/n_ctx padding and F16 DMMV

**Link**: https://github.com/ggml-org/llama.cpp/issues/8798
**State**: closed
**Created**: 2024-07-31T18:54:53+00:00
**Closed**: 2024-08-01T13:26:23+00:00
**Comments**: 3
**Labels**: bug, high severity

### Description

I'm not sure why I can't reproduce this with llama-cli, but I can reproduce it with GPT4All after the merge of PR #7257, up to and including commit 398ede5efeb07b9adf9fbda7ea63f630d476a792 from today (the latest I've tried).

**edit:** I can also reproduce this on commit 952d03dbe from before the padding was increased, so the extra padding for FA seems to have been masking an older bug.

Diagnostic information is given for a fork based on commit 398ede5efeb07b9adf9fbda7ea63f630d476a792, but line numbers won't match exactly in ggml-cuda.cu due to some extra code added for device enumeration, which is required by GPT4All.

cc @slaren @JohannesGaessler

### Steps to reproduce

1. Construct a `llama-2-7b.Q4_0.gguf` model fully offloaded to a single Tesla P40, with n_ctx=2016 (a multiple of 32 but not 256), n_batch=2048, and n_ubatch=512. Flash attention is disabled.
2. In chunks of 128 (the max batch size GPT4All uses in practice), decode 1990 tokens of input.
3. Sample a token

[... truncated for brevity ...]

---

## Issue #N/A: Bug: `illegal hardware instruction` when running on M3 mac Sequoia installed with brew

**Link**: https://github.com/ggml-org/llama.cpp/issues/9676
**State**: closed
**Created**: 2024-09-28T13:33:37+00:00
**Closed**: 2024-11-16T01:59:02+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

When I run `llama-cli -m ./Phi-3.5-mini-instruct-Q6_K_L.gguf -p "I believe the meaning of life is" -n 128` I get an error
```
build: 3829 (44f59b43) with Apple clang version 15.0.0 (clang-1500.3.9.4) for x86_64-apple-darwin23.4.0
main: llama backend init
[1]    36447 illegal hardware instruction  llama-cli -m  -p "I believe the meaning of life is" -n 128
```

I did a fresh install with homebrew, but got the same repeatedly. Seems the same as https://github.com/ggerganov/llama.cpp/issues/8065, but that issue was closed so I wanted to have an open one to track

### Name and Version

```
❯ llama-cli --version
version: 3829 (44f59b43)
built with Apple clang version 15.0.0 (clang-1500.3.9.4) for x86_64-apple-darwin23.4.0

```

### What operating system are you seeing the problem on?

Mac

### Relevant log output

No additional log output available, even with `-v`


---

## Issue #N/A: Bug: CANN: Inference result garbled

**Link**: https://github.com/ggml-org/llama.cpp/issues/10252
**State**: closed
**Created**: 2024-11-11T03:35:51+00:00
**Closed**: 2025-01-13T01:07:32+00:00
**Comments**: 14
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

llama.cpp使用QWen2.5-7b-f16.gg在310P3乱码

### Name and Version

./build/bin/llama-cli -m Qwen2.5-7b-f16.gguf -p "who are you" -ngl 32 -fa

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
build: 4036 (1dc04b2d) with cc (GCC) 10.3.1 for aarch64-linux-gnu
main: llama backend init
main: load the model and apply lora adapter, if any
llama_load_model_from_file: using device CANN0 (Ascend310P3) - 20332 MiB free
llama_load_model_from_file: using device CANN1 (Ascend310P3) - 20331 MiB free
llama_load_model_from_file: using device CANN2 (Ascend310P3) - 20336 MiB free
llama_load_model_from_file: using device CANN3 (Ascend310P3) - 20338 MiB free
llama_load_model_from_file: using device CANN4 (Ascend310P3) - 20339 MiB free
llama_load_model_from_file: using device CANN5 (Ascend310P3) - 20336 MiB free
llama_load_model_from_file: using device CANN6 (Ascend310P3) - 20349 MiB free
llama_model_loader: loaded meta data with 34 ke

[... truncated for brevity ...]

---

## Issue #N/A: Bug:  llama runner process has terminated: GGML_ASSERT(src1t == GGML_TYPE_F32) failed

**Link**: https://github.com/ggml-org/llama.cpp/issues/9902
**State**: closed
**Created**: 2024-10-15T23:43:48+00:00
**Closed**: 2024-11-29T01:09:53+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

I got the following error when running model Imported from GGUF which is generated from the model fine-tuned with LoRA (mlx_lm).

Error: llama runner process has terminated: GGML_ASSERT(src1t == GGML_TYPE_F32) failed

The following are commands used

mlx_lm.lora  --train  --model meta-llama/Llama-3.2-1B  --data ~/Projects/AI/data --iters 1000

mlx_lm.generate --model meta-llama/Llama-3.2-1B --adapter-path ./adapters --prompt "What is biomolecule?"

mlx_lm.fuse --model meta-llama/Llama-3.2-1B --adapter-path ./adapters --export-gguf 

Create Modelfile
FROM ./fused_model/ggml-model-f16.gguf

ollama create example -f Modelfile

ollama run example 

Error: llama runner process has terminated: GGML_ASSERT(src1t == GGML_TYPE_F32) failed
/Users/runner/work/ollama/ollama/llm/llama.cpp/ggml/src/ggml-metal.m:1080: GGML_ASSERT(src1t == GGML_TYPE_F32) failed

### Name and Version

ollama version is 0.3.13

### What operating system are you seeing the problem 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: "GPU + CUDA + VRAM + Shared Memory (UMA)" slower then "CPU + RAM"?

**Link**: https://github.com/ggml-org/llama.cpp/issues/10330
**State**: closed
**Created**: 2024-11-16T06:18:16+00:00
**Closed**: 2025-01-01T01:07:41+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

When forcing llama.cpp to use "GPU + CUDA + VRAM + shared memory (UMA)", we noticed:
- High CPU load (even when only GPU should be used)
- Worse performance than using "CPU + RAM".

More details here: https://github.com/ollama/ollama/issues/7673#issuecomment-2480393630

What could be the reason behind CPU + RAM being faster than GPU + shared memory?
Does llama.cpp prevent "Shared Memory Bank Conflicts"?
See: https://www.microway.com/hpc-tech-tips/gpu-shared-memory-performance-optimization/

### Name and Version

We tested the llama.cpp fork used by ollama version 0.4.1

### What operating system are you seeing the problem on?

Windows

### Relevant log output

_No response_

---

## Issue #N/A: Bug: DLLAMA_VULKAN=1 tag is not linking vulkan

**Link**: https://github.com/ggml-org/llama.cpp/issues/10201
**State**: closed
**Created**: 2024-11-07T14:19:07+00:00
**Closed**: 2024-12-23T01:30:33+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

I am trying to cross compile the llama.cpp with vulkan support targetting an android device which has adreno GPU. While trying with "-DGGML_VULKAN=1" tag I can see that the making process in linking vulkan libraries but I dont need GGML I just need LLAMA_VULKAN. 

It is not linking the vulkan libraries.

**Can recreate the issue using this command:**

cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk-r27c/build/cmake/android.toolchain.cmake \
-DANDROID_ABI=arm64-v8a \
-DANDROID_PLATFORM=android-23 \
-DCMAKE_C_FLAGS="-march=armv8.2-a+fp+simd+crc+crypto" \
-DCMAKE_CXX_FLAGS="-march=armv8.2-a+fp+simd+crc+crypto" \
-DLLAMA_VULKAN=1 \
-DBUILD_SHARED_LIBS=OFF \
-DVulkan_INCLUDE_DIR=/path/to/Vulkan-Headers/include \
-DVulkan_LIBRARY=/path/to/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/33/libvulkan.so \
-DLLAMA_OPENMP=OFF \
-DGGML_LLAMAFILE=OFF \
-B build-android ..

**The output that says it is not linki

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Unable to generate imatrix data using imatrix.exe from release 3089

**Link**: https://github.com/ggml-org/llama.cpp/issues/7765
**State**: closed
**Created**: 2024-06-05T08:49:47+00:00
**Closed**: 2024-06-06T13:30:59+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

Started having this issue today, assuming after the only recent commit to `imatrix.cpp` (https://github.com/ggerganov/llama.cpp/commit/1442677f92e45a475be7b4d056e3633d1d6f813b#diff-75299e7302c2c05622a58cb4901166fc1e94bcdde981961e55e3d939cbf41825), where using this command:

`.\imatrix.exe -m .\model.gguf -f .\imatrix.txt -ngl 10`

...doesn't work anymore, it doesn't error or anything, imatrix generation just doesn't happen.

Am I missing something that was added to this process over the last three days?

It works as expected (imatrix) in the version I used from 3 days ago, namely **[b3070](https://github.com/ggerganov/llama.cpp/releases/tag/b3070), before this change**, but it could have been another one from the commit - or other changes to build.


### Name and Version
[b3089](https://github.com/ggerganov/llama.cpp/releases/tag/b3089)
[llama-b3089-bin-win-cuda-cu12.2.0-x64](https://github.com/ggerganov/llama.cpp/releases/download/b3089/llama-b3089-b

[... truncated for brevity ...]

---

## Issue #N/A: Bug: infill reference crashed

**Link**: https://github.com/ggml-org/llama.cpp/issues/8138
**State**: closed
**Created**: 2024-06-26T13:54:35+00:00
**Closed**: 2024-06-27T07:46:42+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

./llama-infill -t 10 -ngl 0 -m ../../models/Publisher/Repository/codellama-13b.Q3_K_S.gguf --temp 0.7 --repeat_penalty 1.1 -n 20 --in-prefix "def helloworld():\n    print(\"hell" --in-suffix "\n   print(\"goodbye world\")\n    "

this command causes llama.cpp abort

### Name and Version

./llama-llava-cli --version
version: 3235 (88540445)
built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.5.0

### What operating system are you seeing the problem on?

Mac

### Relevant log output

```shell
Log start
main: build = 3235 (88540445)
main: built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.5.0
main: seed  = 1719410022
llama_model_loader: loaded meta data with 20 key-value pairs and 363 tensors from ../../models/Publisher/Repository/codellama-13b.Q3_K_S.gguf (version GGUF V2)
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:

[... truncated for brevity ...]

---

## Issue #N/A: Bug: server crashed today for the first time.

**Link**: https://github.com/ggml-org/llama.cpp/issues/7637
**State**: closed
**Created**: 2024-05-30T11:16:57+00:00
**Closed**: 2024-07-14T01:07:11+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

I created an assistant, it's instructed to output a json object at the end of the chat.
The chat went on perfectly but at the very end the server crashed.

### Name and Version

[built just a few moments ago]
\bin\main --version
version: 3029 (b864b50c)
built with clang version 18.1.5 for x86_64-w64-windows-gnu

Note: I am using the same prompt as usual, and this never happened before.


### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
{"tid":"14964","timestamp":1717067190,"level":"INFO","function":"init","line":715,"msg":"initializing slots","n_slots":1}
{"tid":"14964","timestamp":1717067190,"level":"INFO","function":"init","line":727,"msg":"new slot","id_slot":0,"n_ctx_slot":1536}
{"tid":"14964","timestamp":1717067190,"level":"INFO","function":"main","line":3040,"msg":"model loaded"}
{"tid":"14964","timestamp":1717067190,"level":"INFO","function":"main","line":3065,"msg":"chat template","chat_example":"

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Build failure in master on Ubuntu 24.04 with CUDA enabled

**Link**: https://github.com/ggml-org/llama.cpp/issues/9473
**State**: closed
**Created**: 2024-09-13T15:35:09+00:00
**Closed**: 2024-09-16T14:22:09+00:00
**Comments**: 23
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

Build failure starting ~Sep 11/12.

I run fresh builds periodically - about once every 1-2 days and this started recently. Build command:
make GGML_CUDA=1 -j 16

### Name and Version

Environment is Ubuntu 24.04 updated as of submission.

commit feff4aa8461da7c432d144c11da4802e41fef3cf (HEAD -> master, tag: b3751, origin/master, origin/HEAD)
gcc (Ubuntu 13.2.0-23ubuntu4) 13.2.0

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Aug_14_10:10:22_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0

cuda-toolkit-12-6 is already the newest version (12.6.1-1).


### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
nvcc -std=c++11 -O3 -g -use_fast_math --forward-unknown-to-host-compiler -arch=native -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_MMV_Y=1 -DK_QUANTS_PER_ITERATION=2 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128  -Iggml/include -Iggml

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Server ends up in infinite loop if number of requests in the batch is greater than parallel slots with system prompt

**Link**: https://github.com/ggml-org/llama.cpp/issues/7834
**State**: closed
**Created**: 2024-06-08T10:36:23+00:00
**Closed**: 2024-07-24T01:06:48+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

If we send a batch of requests to `/completion` endpoint with `system_prompt` it get's stuck in infinite waiting because of the way system is updated.


As per my analysis, what happens is if the number of requests in batch are greater than number of available slots. It leads to extra requests getting stored in `deferred` queue.

As soon as even a single slot is available, it will lead to task transferred from defer queue to normal queue and trigger system_update (because system_prompt is available). System_update will change the state of all the other slots from `PROCESSING` -> `IDLE` because of release function.

So, for all the other slots, as they still have next_tokens to generate their final response is not sent and multitask_subtask_result_pending never gets updated.

One of the possible solution is to wait for all slots to finish before `system_update` is called or the other would be call `send_final_response` if slot has has_next_token as true in

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Hang when running llama-cli compiled with GGML_CUDA_FA_ALL_QUANTS=ON

**Link**: https://github.com/ggml-org/llama.cpp/issues/9975
**State**: closed
**Created**: 2024-10-21T08:33:35+00:00
**Closed**: 2024-12-06T01:07:38+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

Hi! I am learning to adjust the precision of KV Cache in llama.cpp, and I have compiled the `llama-cli` using `GGML_CUDA_FA_ALL_QUANTS=ON` following the documents.
However, when I tried to run the `llama-cli` after compiling, the program hangs without any output.
I am not sure if I am doing something wrong or if there is a bug in the code.
Any help would be appreciated!

- The compile command I used:
```bash
cmake -B build_quants -DGGML_CUDA=ON -DGGML_CUDA_FORCE_CUBLAS=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build_quants --config Release
```

- The start command I used:
```bash
./llama.cpp-b3938/build_quants/bin/llama-cli -m ../models/deepseek-llm-7b-base-q4_0.gguf -p "What is the meaning of life?"
```

Then, the program hangs after displaying the following information. It seems not a model issue, because I have tried other models and got the same result.
```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: b3188 breaks row split mode for multiple GPUs

**Link**: https://github.com/ggml-org/llama.cpp/issues/8801
**State**: closed
**Created**: 2024-07-31T22:00:45+00:00
**Closed**: 2024-09-11T08:22:41+00:00
**Comments**: 11
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

Since commit b3188 llama-cli produce incoherent output on multi-gpu system with CUDA and row tensor splitting.
Layer tensor split works fine but is actually almost twice slower.
GPU are 3x Nvidia Tesla + 3090
All future commits seems to be affected.

### Name and Version

llama-cli version b3188 built on Debian 12.

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: Bug: 15 GiB of CPU RAM permanently leaked on each llama-cli invocation

**Link**: https://github.com/ggml-org/llama.cpp/issues/10050
**State**: closed
**Created**: 2024-10-25T19:03:55+00:00
**Closed**: 2024-10-25T19:10:40+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

1. Reboot my machine
2. Run `free -m` to check memory usage - ~4 GiB
3. Run `llama-cli -m Llama-3.2-3B.Q3_K_M.gguf -p "I believe the meaning of life is" -n 128 -fa`
4. Run `free -m` to check memory usage - ~18 GiB
5. Run htop - no application is using that much RAM.
6. Run llama-cli again and free -m reports ~30 GiB of memory used on system

Only way to recover the RAM is to reboot.

The most suspicious thing is `CPU KV buffer size = 14336.00 MiB` in the logs - that's about the amount of RAM.

I suspect this is probably some kind of Nvidia driver bug with a kernel bug being way down the list and a llama.cpp issue being near impossible since the process is dead but I figured I'd report it here since I don't know how to track it down / where upstream to report to.

### Name and Version

```
$ build/bin/llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  

[... truncated for brevity ...]

---

## Issue #N/A: Bug: I use llama-b3091-bin-win-llvm-arm64.zip Run qwen2-0_5b-instruct-q8_0.gguf and it cannot start. Is it a compilation error of llama-b3091-bin-win-llvm-arm64.zip?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7873
**State**: closed
**Created**: 2024-06-11T07:02:50+00:00
**Closed**: 2024-07-27T01:06:46+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

I use llama-b3091-bin-win-llvm-arm64.zip
Run qwen2-0_5b-instruct-q8_0.gguf and it cannot start. Is it a compilation error of llama-b3091-bin-win-llvm-arm64.zip?

### Name and Version

llama-b3091-bin-win-llvm-arm64.zip

### What operating system are you seeing the problem on?

Other? (Please let us know in description)

### Relevant log output

```shell
windows-arm64
```


---

## Issue #N/A: Bug: Inference is messed up in llama-server+default ui and llama-cli but works in llama-server+openweb ui

**Link**: https://github.com/ggml-org/llama.cpp/issues/8027
**State**: closed
**Created**: 2024-06-20T06:25:45+00:00
**Closed**: 2024-06-25T10:23:52+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

Using: https://huggingface.co/bartowski/Hermes-2-Theta-Llama-3-8B-GGUF/blob/main/Hermes-2-Theta-Llama-3-8B-Q6_K.gguf

**llama-cli**
```bash
./llama-cli -m ~/data/models/Hermes-2-Theta-Llama-3-8B-Q6_K.gguf -ngl 99 -ts 1,1 -t 8 -c 4096 --interactive-first
Hello
=====                          

This is a small hello world program written in Java.

Compile                        
=======                        

To compile, simply run the following command:

    javac Hello.java

Run                            
===                            

To run the program, run the following command:

    java Hello                 

This will output:

    Hello, World! 

You can also run the program directly from the source code by using the following command:

    javac Hello.java && java Hello.java
```
this went on and on

**llama-server + default ui**

```bash
./llama-server -m ~/data/models/Hermes-2-Theta-Llama-3-8B-Q6_K.gguf -ngl 99 -ts 1

[... truncated for brevity ...]

---

## Issue #N/A: Bug: convert-hf-to-gguf-update.py breaks in Windows Python 3.11.5

**Link**: https://github.com/ggml-org/llama.cpp/issues/7706
**State**: closed
**Created**: 2024-06-03T02:11:49+00:00
**Closed**: 2024-06-05T17:07:25+00:00
**Comments**: 10
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

Instead of normal completion, the convert-hf-to-gguf-update.py script ran afoul of smaug-bpe with what appears to be a code page issue and halted.

### Name and Version

Current pull on branch master as of a few minutes ago.

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
INFO:convert-hf-to-gguf-update:chkhsh: 27949a2493fc4a9f53f5b9b029c82689cfbe5d3a1929bb25e043089e28466de6
INFO:convert-hf-to-gguf-update:normalizer: null
INFO:convert-hf-to-gguf-update:pre_tokenizer: {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": true
}
INFO:convert-hf-to-gguf-update:
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO:convert-hf-to-gguf-update:model: smaug-bpe
INFO:convert-hf-to-gguf-update:tokt: 2
INFO:convert-hf-to-gguf-update:repo: https://huggingface.co/abacusai/Smaug-Llama-3-70B-Instr

[... truncated for brevity ...]

---

## Issue #N/A: Bug: GGML can no longer be statically linked to llama.cpp due to the source code reorganization

**Link**: https://github.com/ggml-org/llama.cpp/issues/8166
**State**: closed
**Created**: 2024-06-27T12:24:37+00:00
**Closed**: 2024-06-30T11:01:04+00:00
**Comments**: 9
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

Since the commit #8006 GGML is now compiled as Dynamic library (vs static library, before).

I can't find any option to reintroduce the previous mode. There is a GGML_STATIC option into the CMakeLists.txt of the ggml solution but it seems to do nothing.

I there a way to reintroduce static compilation mode?

Thanks a lot!

Loïc

### Name and Version

latest.

cmake .. -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=ON -DGGML_AVX2=ON -DGGML_AVX512=OFF -DGGML_STATIC=ON

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Bug: RPC backend crash

**Link**: https://github.com/ggml-org/llama.cpp/issues/9214
**State**: closed
**Created**: 2024-08-28T04:34:25+00:00
**Closed**: 2024-08-30T06:08:39+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

crash

### Name and Version

# ./build_rpc_cuda/bin/llama-cli --version
version: 3639 (20f1789d)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
# CUDA_VISIBLE_DEVICES=0  ./build_rpc_cuda/bin/rpc-server -p 50052 -H 0.0.0.0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Host ('0.0.0.0') is != '127.0.0.1'
         Never expose the RPC server to an open network!
         This is an experimental feature and is not secure!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

create_backend: using CUDA backend
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes
Starting RPC server on 0.0.0.0:50052, backend memory: 7686 MB
Accepted client connection, fre

[... truncated for brevity ...]

---

## Issue #N/A: Bug: In a small n_ctx_slot, the llama.cpp begins gibberish

**Link**: https://github.com/ggml-org/llama.cpp/issues/8498
**State**: closed
**Created**: 2024-07-16T04:28:05+00:00
**Closed**: 2024-07-17T07:33:00+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

While testing the Deepseek Coder V2 Lite(https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) with ollama's default settings(n_ctx 2048), I noticed that the llama.cpp was printing meaningless sentences
After clone the latest llama.cpp, I tested the -c and -np options and found that the model breaks the moment the llama.cpp shifts the slot context at a small ctx length (perhaps if the ctx length is less than about 8192)

### Name and Version

version: 3400 (97bdd26e)
built with cc (Ubuntu 13.2.0-23ubuntu4) 13.2.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
llama-server ... -c 16000 -np 4, llama-server ... -c 4000 -np 1, llama-server ... -c 6000 -np 1

As shown below, after the n_keep = 1 log occurs, the model prints garbage infinitely ...

slot context shift | .... n_keep = 1 n_left = 3998 n_discard = 1999 n_ctx=4000 n_past=3999 n_system_tokens=0 n_cache_t

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Ccache causing SYCL backend failed to build on Windows

**Link**: https://github.com/ggml-org/llama.cpp/issues/9954
**State**: closed
**Created**: 2024-10-19T23:26:25+00:00
**Closed**: 2024-12-25T01:07:24+00:00
**Comments**: 9
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

With ccache installed in Windows, trying to build with SYCL by following instructions in https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/SYCL.md#windows.
I found that *.cpp.obj files were unable to be placed in right location but in having them with .o extension in <project root>/build folder. This behavior causing linker failed to do the job.

Here is the screenshot:
![image](https://github.com/user-attachments/assets/3a3d4376-b2ad-44c9-9633-bd8203947ef4)


### Name and Version

llama.cpp repository cda0e4b648dde8fac162b3430b14a99597d3d74f
Windows 11 23H2 10.0.22631.4317
VS2022 Community 17.11.5
Intel oneAPI 2024.2.1
MSVC v143
Ccache 4.10.2

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
[1/184] Building CXX object src\CMakeFiles\llama.dir\unicode-data.cpp.obj
[2/184] Building CXX object common\CMakeFiles\build_info.dir\build-info.cpp.obj
[3/184] Building CXX object common\CMakeFiles\

[... truncated for brevity ...]

---

