# router - issues

**Total Issues**: 12
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 6
- Closed Issues: 6

### Label Distribution

- router: 12 issues
- inactive: 4 issues
- enhancement: 2 issues
- bug: 1 issues
- high priority: 1 issues
- collaboration: 1 issues
- feature: 1 issues

---

## Issue #N/A: [Router]  [Performance] SGLang hits connection limit at ~32k concurrent requests, causing failures

**Link**: https://github.com/sgl-project/sglang/issues/7551
**State**: open
**Created**: 2025-06-26T06:19:16+00:00
**Comments**: 0
**Labels**: bug, router

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

## Summary

SGLang experiences severe performance degradation and request failures when concurrent connections approach 32,768 (2^15), strongly indicating a file descriptor limit issue. While the router provides better resilience than direct worker connections, both modes eventually fail at this threshold. We hope to fully solve this issue

[... truncated for brevity ...]

---

## Issue #N/A: Task 000: Centralized Configuration Module

**Link**: https://github.com/sgl-project/sglang/issues/7533
**State**: closed
**Created**: 2025-06-25T20:09:47+00:00
**Closed**: 2025-06-27T22:42:03+00:00
**Comments**: 0
**Labels**: enhancement, router

### Description

# Task 000: Centralized Configuration Module

## Summary
Create a comprehensive configuration module that centralizes all validation logic, provides type-safe configuration structures, and eliminates scattered validation code throughout the router.

## Problem Statement
Currently, configuration validation is scattered across multiple locations:
- URL validation happens in Python code
- Mode compatibility checks occur during server startup
- Policy parameter validation is embedded in individual routers
- No centralized error handling for configuration issues
- Duplicate validation logic in different components

This leads to:
- Inconsistent validation rules
- Runtime errors that could be caught at startup
- Difficult maintenance when adding new configuration options
- Poor error messages that don't guide users to fixes

## Proposed Solution

### 1. Configuration Type System
Create strongly-typed configuration structures with built-in validation:

```rust
// src/config/types.rs
#[derive(

[... truncated for brevity ...]

---

## Issue #N/A: SGLang Router Architecture Improvement Proposal

**Link**: https://github.com/sgl-project/sglang/issues/7532
**State**: open
**Created**: 2025-06-25T20:06:12+00:00
**Comments**: 1
**Labels**: high priority, collaboration, router

### Description

# SGLang Router Architecture Improvement Proposal

## Table of Contents
1. [Summary](#summary)
2. [Current Architecture Overview](#current-architecture-overview)
3. [System Components](#system-components)
4. [Request Flow Analysis](#request-flow-analysis)
5. [Identified Pain Points](#identified-pain-points)
6. [Proposed Improvements](#proposed-improvements)
7. [Long-Term Vision](#long-term-vision)
8. [Implementation Phases](#implementation-phases)
9. [Risk Analysis](#risk-analysis)
10. [Success Metrics](#success-metrics)
11. [Conclusion](#conclusion)
12. [Appendix: Architecture Diagrams](#appendix-architecture-diagrams)

## Summary

This proposal outlines a architectural improvement plan for the SGLang Router, a high-performance load balancer that supports both traditional and disaggregated (Prefill-Decode) routing modes. The improvements focus on enhancing maintainability and extensibility without disrupting existing functionality. These changes lay the foundation for a long-term tran

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Router Crashes Intermittently

**Link**: https://github.com/sgl-project/sglang/issues/6491
**State**: open
**Created**: 2025-05-21T07:51:23+00:00
**Comments**: 0
**Labels**: router

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

In our scenario, the router inexplicably crashes after a period of time (which could range from hours to weeks). Through the logs, I identified the following errors:
```
{"log":"[Router (Rust)] 2025-05-21 03:52:00 - DEBUG - starting new connection: http://192.168.0.99:8001/\n","stream":"stderr","time":"2025-05-21T03:52:00.40688383Z"}
{"log

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support for batch inference within sglang-router over multi-nodes

**Link**: https://github.com/sgl-project/sglang/issues/6446
**State**: open
**Created**: 2025-05-20T05:16:50+00:00
**Comments**: 5
**Labels**: router

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently only the sglang-router supports data parallelism over multiple nodes. However the sglang-router does not provide HTTP APIs such as `/v1/files `and `/v1/batches`. And there is currently no offline engine supporting data parallelism over multiple nodes, this limits the ability to perform large-scale batch inference in distributed setups. We believe adding these APIs to sglang-router would be a valuable addition.

Any consideration or guidance on implementing this feature would be greatly appreciated.


### Related resources

[HTTP APIs in sglang http server](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)

[HTTP APIs in sglang-router http server](https://github.

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] When using sgl-router, canceled requests keep running on workers.

**Link**: https://github.com/sgl-project/sglang/issues/6280
**State**: open
**Created**: 2025-05-14T05:57:23+00:00
**Comments**: 1
**Labels**: router

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I started sglang router with one worker under it, sent a /chat/completions request, canceled it before response, and it continued running on the worker (checked via nvidia-smi and sglang logs). In fact I believe it was stuck running - didn't finish after 5 minutes.

### Reproduction

```sh
python -m sglang.launch_server \\
    --model-path

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] sglang-router curl get return without content-type: application/json in the header (#3307 reopened)

**Link**: https://github.com/sgl-project/sglang/issues/6237
**State**: open
**Created**: 2025-05-12T13:54:05+00:00
**Comments**: 3
**Labels**: router

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I want to reopen (https://github.com/sgl-project/sglang/issues/3307) as the described bug below is not solved in the current version. The bug prohibits the use of sglang in combination with OpenWebUI (https://github.com/open-webui/open-webui).


**Reopened from #3307**

Thanks for this wonderful router. We are trying it to add several sgla

[... truncated for brevity ...]

---

## Issue #N/A: how to update weight with sglang_router? Or how to get worker_urls

**Link**: https://github.com/sgl-project/sglang/issues/4282
**State**: closed
**Created**: 2025-03-11T03:57:17+00:00
**Closed**: 2025-05-18T00:20:51+00:00
**Comments**: 4
**Labels**: inactive, router

### Description

No description provided.

---

## Issue #N/A: [Bug] sglang-router failure when first load model, try again successed

**Link**: https://github.com/sgl-project/sglang/issues/4160
**State**: closed
**Created**: 2025-03-07T04:33:47+00:00
**Closed**: 2025-06-11T00:19:41+00:00
**Comments**: 3
**Labels**: inactive, router

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When first serving a model, sometimes it fails, but when I serve it again, it works.

I inspected the failed log and found that all 8 workers say 'The server is fired up and ready to roll!', but the router says 'health check is pending with error: error sending request for url (http://127.0.0.1:31000/health)' after 600 seconds.

### Reprod

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Can router support prometheus metrics

**Link**: https://github.com/sgl-project/sglang/issues/3393
**State**: closed
**Created**: 2025-02-08T06:42:46+00:00
**Closed**: 2025-04-28T00:19:29+00:00
**Comments**: 3
**Labels**: enhancement, inactive, feature, router

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

K8s is often used to deploy applications online. After the router module is introduced, related service indicator monitoring is also required. Therefore, similar to https://github.com/sgl-project/sglang/pull/1853 provided by the server, does it support the collection of monitoring indicators of the router?

### Related resources

_No response_

---

## Issue #N/A: [Feature] Support service discovery on Kubernetes in router

**Link**: https://github.com/sgl-project/sglang/issues/3073
**State**: closed
**Created**: 2025-01-23T07:08:03+00:00
**Closed**: 2025-03-26T00:17:50+00:00
**Comments**: 3
**Labels**: inactive, router

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

This feature proposes adding Kubernetes service discovery support to the router component. Service discovery will enable the router to dynamically identify and connect to backend services running in a Kubernetes cluster. This is particularly useful for distributed systems where backend instances may scale up or down dynamically.

## UI/UX

```bash
# New approach
python -m sglang_router.launch_router --worker-service-on-k8s default/sglang-svc
# Static approach
python -m sglang_router.launch_router --worker-urls http://worker_url_1 http://worker_url_2
```

## Pseudo code

```py
# Load Kubernetes configuration (e.g., from kubeconfig or in-cluster config)
load_kube_config()

# Initialize Kubernetes API client
api_clien

[... truncated for brevity ...]

---

## Issue #N/A: Can router support --api-key parameter

**Link**: https://github.com/sgl-project/sglang/issues/3031
**State**: closed
**Created**: 2025-01-21T10:02:21+00:00
**Closed**: 2025-01-24T04:30:32+00:00
**Comments**: 4
**Labels**: router

### Description

When I add an api key to the worker, the router cannot access it

---

