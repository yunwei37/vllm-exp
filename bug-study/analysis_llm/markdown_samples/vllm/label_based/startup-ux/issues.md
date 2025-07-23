# startup-ux - issues

**Total Issues**: 7
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 7
- Closed Issues: 0

### Label Distribution

- startup-ux: 7 issues
- RFC: 4 issues
- feature request: 3 issues
- torch.compile: 2 issues

---

## Issue #N/A: [RFC]: vLLM-compile low-hanging fruit cold start improvements

**Link**: https://github.com/vllm-project/vllm/issues/20451
**State**: open
**Created**: 2025-07-03T19:22:18+00:00
**Comments**: 0
**Labels**: RFC, torch.compile, startup-ux

### Description

### Motivation.

This issue tracks potential low-hanging fruit for improving vLLM-compile cold start time. @anijain2305, @BoyuanFeng, and I sat down to look at some traces and noticed some things we can improve.

There are more longer-term projects for improving torch.compile cold start time, but those will probably take a bit to hit.

### Proposed Change.

- [ ] vLLM's [custom bytecode hook](https://github.com/vllm-project/vllm/blob/536fd330036b0406786c847f68e4f67cba06f421/vllm/compilation/wrapper.py#L77-L121) seems to take a long time (~7 seconds on llama-3.1-70b model). I'm not sure how much of this is actually needed for runtime execution. We should guard the decompilation step behind an envvar. If VLLM_COMPILE_DEPYF=0 (default), we write out a `transformed_code.py` that has a comment that says "Please set VLLM_COMPILE_DEPYF=1 to populate this file".
- [ ] In llama-3.1-70b, with piecewise cudagraphs, we split a module into 80 different subgraphs. A lot of these subgraphs are litera

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: vLLM-compile (minus cudagraphs) warm-start time should be close to zero

**Link**: https://github.com/vllm-project/vllm/issues/20402
**State**: open
**Created**: 2025-07-02T22:04:20+00:00
**Comments**: 6
**Labels**: RFC, torch.compile, startup-ux

### Description

### Motivation.

@BoyuanFeng did some benchmarks of vLLM cold vs warm start of a 70B model. In the warm start, compilation (ignoring cudagraphs) took 25 out of 132 seconds, almost 20% of the time. On warm start, all of the hard work (compiling artifacts) should have been already done.

The theoretical minimum amount of time that vLLM-compile needs to spend in warm start is the amount of time it takes to load all the compiled code.

![Image](https://github.com/user-attachments/assets/b34204f8-5ad5-49d4-bdc6-6805610ac6be)

### Proposed Change.

The following categories correspond to what is in the chart above.

Dynamo:
- On warm start, vLLM always re-runs Dynamo. We don't need to do this: instead, we can directly serialize the bytecode that Dynamo produces and re-load it.
- Originally I was planning on waiting until torch.compile implemented "precompilation", which will skip Dynamo on warm start. It might be worth figuring out how to get a simpler version of this into vLLM, especially be

[... truncated for brevity ...]

---

## Issue #N/A: [RFC][UX][torch.compile][CUDAGraph]: Overhaul `CompilationConfig` and improve CLI `-O<n>`

**Link**: https://github.com/vllm-project/vllm/issues/20283
**State**: open
**Created**: 2025-06-30T21:47:04+00:00
**Comments**: 7
**Labels**: RFC, startup-ux

### Description


**tl;dr**: Improve the user experience around compilation and cudagraph capture by consolidating/overhauling `CompilationConfig` and defining more meaningful optimization levels `-O0`, `-O1`, `-O2`, `-O3` (and maybe more).

## Motivation.

`CompilationConfig` was born around December 2024 to enable configuring `torch.compile`-based compilation and piecewise cudagraph capture. Since then, a bunch more flags were added to support new features, all good in isolation but without a cohesive plan. As vLLM aims to provide great performance out-of-the-box, having to manually configure a bunch of flags is bad UX.

`CompilationConfig` currently serves as both the user-facing and compiler-interfacing compilation configuration mechanism. What I mean by that is that it's used by CLI/Python API users to control compilation, as well as other parts of the codebase (model runner, vllm config, etc.). This has the benefit of good UX for developers to directly control compilation from the CLI and Python,

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Lazy CUDA Graph capture

**Link**: https://github.com/vllm-project/vllm/issues/20098
**State**: open
**Created**: 2025-06-25T21:27:51+00:00
**Comments**: 11
**Labels**: RFC, startup-ux

### Description

### Motivation.

Currently vLLM captures cudagraphs as part of the engine initialization significantly slowing down vLLM startup time. By default, vLLM captures 66 graphs, which depending on model size and GPU type, can take more than 10s. This is not great UX (see #19824 for details).

In addition, It's most unlikely that all 66 graphs are actually needed, wasting both time and space.  

### Proposed Change.

We propose to capture cudagraphs lazily. Instead of performing dummy runs during the engine initialization phase, the idea is to do those runs somewhere in the CUDA piecewise backend, and only for the current runtime shape if not cached already.

Exact implementation needs to be worked out.

### Feedback Period.

one week

### CC List.

@ProExpertProg @aarnphm @charlesfrye  

### Any Other Things.

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documenta

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Improve startup time UX

**Link**: https://github.com/vllm-project/vllm/issues/19824
**State**: open
**Created**: 2025-06-19T00:41:47+00:00
**Comments**: 8
**Labels**: feature request, startup-ux

### Description

# 🚀 The feature, motivation and pitch

vLLM startup time has become a pain-point for certain use cases, like auto-scaling instances or model swapping. This leads to poor user experience or even users choosing to use `--enforce-eager`, sacrificing performance. I'm creating this parent issue to track our work on better understanding the startup time as well as the throughput tradeoffs from skipping certain steps.

## 🚧 [WIP] Startup heavy hitters (most time-consuming)

1. P2P access check
2. Weight loading
3. Dynamo tracing
4. Inductor compilation
  a. Additional time spent on extra compile_sizes and max-autotune
5. CUDAGraph capture
6. PTX compilation

### Other
- @ywang96 mentioned LMMs take a long time generating dummy multi-modal data in the profile_run
- 

Recent measurements from @robertgshaw2-redhat:
> Llama-70B-Fp8 on TP=8. I see the following:
> - ~60s to check for P2P access manually. We can disable this check with VLLM_SKIP_P2P_CHECK=1
> - ~10s to load weights --- from hot pag

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: `CustomOp` cleanup

**Link**: https://github.com/vllm-project/vllm/issues/19817
**State**: open
**Created**: 2025-06-18T20:10:27+00:00
**Comments**: 0
**Labels**: feature request, startup-ux

### Description

### 🚀 Motivation

Currently, we do not have a consistent plan for "light" custom ops (subclasses of `CustomOp` with both torch-native and GPU implementations). As we work on improved performance on NVIDIA Blackwell and AMD, we should be more intentional with CompilationConfig defaults that control custom op dispatching. This is a parent issue that tracks smaller PRs addressing `CustomOp`s.

In vLLM, there are two kinds of custom kernels/ops:
1. "heavy" ops like GEMMs, MoE, and attention, which will mostly use tuned custom kernels for maximum performance.
2. "light" ops like `RMSNorm`, `SiluAndMul`, and `RoPE`, which have both torch-native and custom GPU implementations.

This issue only refers to "light" ops, which are (or should be) all subclasses of `CustomOp`.

When we enabled `torch.compile` by default in V1, the plan was to reduce our reliance on custom kernels to reduce maintenance costs and code complexity, even with minor performance costs. Recent versions of `torch` actually p

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Add opentelemetry tracing for vLLM start up phases

**Link**: https://github.com/vllm-project/vllm/issues/19318
**State**: open
**Created**: 2025-06-07T16:10:23+00:00
**Comments**: 2
**Labels**: feature request, startup-ux

### Description

### 🚀 The feature, motivation and pitch

This FR asks for tracing through vLLM cold starts. This would include key phases, as trace spans, leading up to the FastAPI HTTP server is up and running. #17794 is related but asks for tracing requests.

Why would this be useful?

* To facilitate cold start optimizations, both for vLLM users and contributors.
  This is important for quick auto scaling of inference workloads in cloud
  environments.
* Users may want to tweak vLLM settings based on which phase is contributig to
  high latency, e.g. changing how the model is loaded using `--load-format
  runai_streamer`.
* Contributors interested in performance optimization need this data to know
  which area to focus on and the visual phase breakdown provided by traces is
  much easier to interpret quickly than logs. This is how I noticed #19317.

The set of key spans and their attributes could be iterated on over time but I
think it'd be interesting to include at least

* Python import time (whi

[... truncated for brevity ...]

---

