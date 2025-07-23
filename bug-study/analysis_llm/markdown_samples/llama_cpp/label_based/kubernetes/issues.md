# kubernetes - issues

**Total Issues**: 2
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 1

### Label Distribution

- server/webui: 2 issues
- kubernetes: 2 issues
- enhancement: 1 issues
- help wanted: 1 issues
- bug-unconfirmed: 1 issues

---

## Issue #N/A: kubernetes example

**Link**: https://github.com/ggml-org/llama.cpp/issues/6546
**State**: open
**Created**: 2024-04-08T16:31:37+00:00
**Comments**: 18
**Labels**: enhancement, help wanted, server/webui, kubernetes

### Description

### Motivation

Kubernetes is widely used in the industry to deploy product and application at scale.

It can be useful for the community to have a `llama.cpp` [helm](https://helm.sh/docs/intro/quickstart/) chart for the server.

I have started several weeks ago, I will continue when I have more time, meanwhile any help is welcomed:

https://github.com/phymbert/llama.cpp/tree/example/kubernetes/examples/kubernetes

### References
- #6545


---

## Issue #N/A: server-cuda closes connection while still processing tasks

**Link**: https://github.com/ggml-org/llama.cpp/issues/6545
**State**: closed
**Created**: 2024-04-08T16:09:49+00:00
**Closed**: 2024-04-08T20:34:17+00:00
**Comments**: 5
**Labels**: server/webui, bug-unconfirmed, kubernetes

### Description

Issue to be published in the llama.cpp github: 


I am using the Docker Image ghcr.io/ggerganov/llama.cpp:server-cuda to deploy the server in a Kubernetes cluster in AWS using four A10G gpus. This is the configuration setup: 

>     - name: llama-cpp-server
>         image: ghcr.io/ggerganov/llama.cpp:server-cuda
>         args:
>         - "--model"
>         - "/models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"
>         - "--port"
>         - "8000"
>         - "--host"
>         - "0.0.0.0"
>         - "--ctx-size"
>         - "100000"
>         - "--n-gpu-layers"
>         - "256"
>         - "--cont-batching"
>         - "--parallel" 
>         - "10"
>         - "--batch-size"
>         - "4096"

(not sure if it adds context, but I'm using a persistentVolumeClaim where I download and persist the model)

I already reviewed the server readme and all the command line options and also tested different image tags for server-cuda from the past days. 

Based on

[... truncated for brevity ...]

---

