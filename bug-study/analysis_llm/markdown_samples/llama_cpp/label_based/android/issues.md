# android - issues

**Total Issues**: 5
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 4

### Label Distribution

- android: 5 issues
- build: 2 issues
- bug: 1 issues
- high severity: 1 issues
- stale: 1 issues
- need more info: 1 issues

---

## Issue #N/A: Bug: abort on Android (pixel 8 pro)

**Link**: https://github.com/ggml-org/llama.cpp/issues/8109
**State**: open
**Created**: 2024-06-25T11:28:31+00:00
**Comments**: 9
**Labels**: bug, android, high severity

### Description

### What happened?

![Screenshot_20240625-122328](https://github.com/ggerganov/llama.cpp/assets/26687662/494ac6f9-4467-49a0-a135-1de7bc9ef2f7)

Getting `GGML_ASSERT: ggml.c:21763: svcntb() == QK8_0` when trying to build and run llama.cpp with llama 3 8b q4 on Android (pixel 8 pro)

### Name and Version

version: 3220 (f702a90e)
built with clang version 18.1.7 for aarch64-unknown-linux-android28

### What operating system are you seeing the problem on?

Other? (Please let us know in description)

### Relevant log output

```shell
./llama-server -m ./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --chat-template llama3 --no-mmap
INFO [                    main] build info | tid="542774631704" timestamp=1719314594 build=3220 commit="f702a90e"
GGML_ASSERT: ggml.c:21763: svcntb() == QK8_0
Aborted```
```


---

## Issue #N/A: Illegal instruction on Android (Honor Magic 5)

**Link**: https://github.com/ggml-org/llama.cpp/issues/3622
**State**: closed
**Created**: 2023-10-14T01:54:42+00:00
**Closed**: 2023-10-19T22:06:01+00:00
**Comments**: 18
**Labels**: build, android

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Current Behavior

```
$ git clone --depth 1 https://github.com/ggerganov/llama.cpp
$ cd llama.cpp
$ mkdir -p build
$ rm -rf build/*
$ cd build
$ cmake .. -DLLAMA_SANITIZE_ADDRESS=ON && cmake --build . --config Debug
```
```
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- ARM detected
-

[... truncated for brevity ...]

---

## Issue #N/A: [User] Android build fails with "ld.lld: error: undefined symbol: clGetPlatformIDs"

**Link**: https://github.com/ggml-org/llama.cpp/issues/3525
**State**: closed
**Created**: 2023-10-07T12:03:41+00:00
**Closed**: 2023-10-07T15:12:36+00:00
**Comments**: 17
**Labels**: build, android

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I am trying to use [this tutorial](https://github.com/ggerganov/llama.cpp#building-the-project-using-termux-f-droid) to compile llama.cpp.

# Current Behavior

Compilation failed.

# Environment and Context

Please provide detailed information about your compu

[... truncated for brevity ...]

---

## Issue #N/A: Running llama.cpp on android just prints out the question

**Link**: https://github.com/ggml-org/llama.cpp/issues/712
**State**: closed
**Created**: 2023-04-02T14:16:00+00:00
**Closed**: 2024-04-11T01:07:11+00:00
**Comments**: 2
**Labels**: android, stale

### Description

I ran llama.cpp on my android phone which has 8 threads and 8GB of ram in which around 7.16 GB is available, that is more than enough to run the 7B Alpaca model on it. But when i run it, it just repeats the question that i provided to it. I am using the `./examples/chat.sh` file. Why does it do that? How do i solve it?

---

## Issue #N/A: illegal instructions error on Android

**Link**: https://github.com/ggml-org/llama.cpp/issues/402
**State**: closed
**Created**: 2023-03-22T17:33:25+00:00
**Closed**: 2023-07-28T19:38:12+00:00
**Comments**: 29
**Labels**: need more info, android

### Description

first thanks for the wonderful works so far !!!

i manged to compile it in Linux and windows but i have a problem with android.
i have A52 6 GB but i get "illegal instructions" error.

i compiled the source using wsl2 with  ndk r25 without any errors. i moved the llama folder from sd card to "home" directory in (Termux) in order to have the execute command working. and i converted to original model using the newer source code to avoid "too old" error message but at the end i get this error.

i believe it is because of having avx, avx2 and other instruction already enabled in my build which is arm processors cant handle them but i cant figure it out how to change it to get it working on my android device.
thanks in advanced <3
![ScreenshotTermux](https://user-images.githubusercontent.com/128628434/226988980-5d1a67c3-797b-4eed-8449-164b0c9abefb.jpg)


---

