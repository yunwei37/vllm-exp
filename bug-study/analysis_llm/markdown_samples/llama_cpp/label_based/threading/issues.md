# threading - issues

**Total Issues**: 5
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 5

### Label Distribution

- threading: 5 issues
- enhancement: 4 issues
- stale: 3 issues
- performance: 1 issues
- macos: 1 issues
- hardware: 1 issues
- windows: 1 issues

---

## Issue #N/A: Feature Request: Tensor Parallelism support

**Link**: https://github.com/ggml-org/llama.cpp/issues/9086
**State**: closed
**Created**: 2024-08-19T01:38:13+00:00
**Closed**: 2024-12-13T01:07:40+00:00
**Comments**: 9
**Labels**: enhancement, threading, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Tensor parallelism is a a critical technique employed to train and inference from very large language models by splitting the actual computations/tensors across multiple compute devices. 

### Motivation

In our previous implementation on Xeon CPU, tensor parallelism(TP) can significantly reduce the latency on inference. <html xmlns:v="urn:schemas-microsoft-com:vml"
xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:x="urn:schemas-microsoft-com:office:excel"
xmlns="http://www

[... truncated for brevity ...]

---

## Issue #N/A: CPU performance bottleneck(?) when using macOS Accelerate

**Link**: https://github.com/ggml-org/llama.cpp/issues/5417
**State**: closed
**Created**: 2024-02-08T16:53:12+00:00
**Closed**: 2024-02-11T19:12:45+00:00
**Comments**: 5
**Labels**: enhancement, performance, macos, threading

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

I've been doing some performance testing of llama.cpp in macOS (On M2 Ultra 24-Core) and was comparing the CPU performance of inference with various options, and ran into a very large performance drop - Mixtral model inference on 16 cores (16 because it's only the p

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

## Issue #N/A: [Suggestion] Add parameter for setting openblas threads

**Link**: https://github.com/ggml-org/llama.cpp/issues/1188
**State**: closed
**Created**: 2023-04-26T13:24:17+00:00
**Closed**: 2024-04-09T01:09:52+00:00
**Comments**: 1
**Labels**: enhancement, threading, stale

### Description

Openblas deafults to some maximum available threads, but would probably not be the most optimal.
In Openblas there is a function to set the number of threads, why not use this?

```void openblas_set_num_threads(int num_threads);```

Current workaround is to set an openblas environment variable.

---

## Issue #N/A: [User] Deadlock if number of threads > number of (hyper)threads

**Link**: https://github.com/ggml-org/llama.cpp/issues/1159
**State**: closed
**Created**: 2023-04-24T19:02:29+00:00
**Closed**: 2023-05-03T18:30:11+00:00
**Comments**: 4
**Labels**: threading

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I expect the program to run suboptimally but finish.

# Current Behavior

Currently the program locks up with very large cpu utilization.

# Environment and Context

I have a 6 core intel machine i.e. 12 threads with hyperthreading. 
Once I run with -t 13

[... truncated for brevity ...]

---

