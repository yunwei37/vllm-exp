# Production Bug Taxonomy for LLM Serving Frameworks

## Overview

This document categorizes production bugs found across vLLM, llama.cpp, and SGLang based on analysis of 12,349 GitHub issues.

## 1. Memory Management Issues

### 1.1 Out-of-Memory (OOM) Errors
**Frequency**: High (44 in vLLM, 52 in llama.cpp)
**Impact**: Service termination, request failures

**Common Patterns**:
- Large batch sizes exceeding VRAM capacity
- Memory leaks during long-running inference
- Fragmentation from dynamic allocation

**Example**: vLLM #21170 - "KV cache timeout leading to memory leak"
```
If the kv_cache for storing a request times out and no exception 
handling is performed, it results in the request being unable to 
proceed with inference and the block being unable to be released.
```

### 1.2 Memory Leaks
**Characteristics**:
- Gradual memory consumption increase
- Often related to improper cleanup of tensors
- Exacerbated by high request rates

## 2. Concurrency & Synchronization Bugs

### 2.1 Race Conditions
**Frequency**: Very High (396 in vLLM, 233 in llama.cpp)

**Types**:
- KV cache corruption when multiple threads access
- Request scheduling conflicts
- Shared resource contention

**Example**: vLLM #21175 - "Wrongly reuse KV for multimodal input"
- Occurs in prefix-decode disaggregation
- Results in incorrect output generation

### 2.2 Deadlocks
**Scenarios**:
- Multi-GPU synchronization
- Distributed inference coordination
- Request queue management

## 3. GPU/CUDA Issues

### 3.1 CUDA Errors
**Frequency**: High (430 in vLLM, 383 in llama.cpp)

**Common Errors**:
- `CUDA_ERROR_OUT_OF_MEMORY`
- `CUDA_ERROR_ILLEGAL_ADDRESS`
- `NCCL timeout` in multi-GPU setups

### 3.2 GPU Memory Management
**Issues**:
- Fragmentation with dynamic batching
- Incorrect memory pool sizing
- Cross-GPU memory transfers

## 4. API & Protocol Issues

### 4.1 Request Handling
**Frequency**: Highest (497 in vLLM, 470 in llama.cpp)

**Problems**:
- Timeout handling failures
- Malformed request parsing
- Streaming response corruption

**Example**: vLLM #21156 - "9-digit integers truncated to 6 digits in streaming"
- Affects function calling results
- JSON parsing precision loss

### 4.2 Protocol Compliance
**Issues**:
- OpenAI API compatibility gaps
- Streaming protocol violations
- Content-Type mismatches

## 5. Model-Specific Bugs

### 5.1 Model Loading Failures
**Frequency**: High (457 in vLLM, 406 in llama.cpp)

**Causes**:
- Unsupported model architectures
- Quantization incompatibilities
- Weight corruption during loading

### 5.2 Inference Errors
**Types**:
- Structured output generation failures
- Tool calling crashes
- Context length violations

**Example**: vLLM #21148 - "Server hang with gemma-3-27b and structured decoding"
```
Failed to advance FSM for request chatcmpl-78eca22187e24416a39e4c12c73dad75 
for tokens 0. Please file an issue.
```

## 6. Performance Degradation

### 6.1 Latency Issues
**Frequency**: Medium (155 in vLLM, 92 in llama.cpp)

**Causes**:
- Inefficient batching strategies
- Cache misses
- Suboptimal kernel selection

### 6.2 Throughput Bottlenecks
**Patterns**:
- Request queuing delays
- GPU underutilization
- Memory bandwidth saturation

## 7. Scaling & Distribution Issues

### 7.1 Multi-GPU Problems
**Frequency**: Medium (184 in vLLM, 153 in llama.cpp)

**Types**:
- NCCL communication failures
- Load balancing inefficiencies
- Tensor parallelism bugs

### 7.2 Cluster Coordination
**Issues**:
- Node failure handling
- State synchronization
- Network timeouts

## 8. Framework-Specific Patterns

### 8.1 vLLM-Specific
- **Prefix caching bugs**: Cache invalidation issues
- **PagedAttention errors**: Block allocation failures
- **Continuous batching**: Request scheduling conflicts

### 8.2 llama.cpp-Specific
- **C++ memory issues**: Manual memory management errors
- **Platform compatibility**: OS-specific failures
- **Quantization bugs**: Precision loss in specific formats

### 8.3 SGLang-Specific
- **RadixAttention issues**: Prefix tree corruption
- **Constraint decoding**: FSM state errors
- **Frontend-backend sync**: Communication protocol bugs

## Root Cause Analysis

### Primary Root Causes:
1. **Complex State Management** (35%)
   - Distributed state synchronization
   - Cache coherency
   - Request lifecycle management

2. **Resource Constraints** (25%)
   - Limited GPU memory
   - Memory fragmentation
   - Bandwidth limitations

3. **Concurrency Complexity** (20%)
   - Race conditions in shared resources
   - Improper synchronization primitives
   - Lock contention

4. **Integration Issues** (15%)
   - Model compatibility
   - API protocol mismatches
   - Library version conflicts

5. **Edge Cases** (5%)
   - Unusual input patterns
   - Rare model architectures
   - Extreme load conditions

## Mitigation Strategies

### Immediate Actions:
1. Implement comprehensive error handling
2. Add request validation layers
3. Improve resource monitoring

### Long-term Solutions:
1. Redesign state management systems
2. Implement formal verification for critical paths
3. Develop automated testing frameworks

## Severity Classification

### Critical (P0):
- Service crashes
- Data corruption
- Security vulnerabilities

### High (P1):
- Performance degradation >50%
- Frequent request failures
- Memory leaks

### Medium (P2):
- Intermittent errors
- Performance degradation <50%
- Feature limitations

### Low (P3):
- Edge case failures
- Minor inconsistencies
- Documentation gaps