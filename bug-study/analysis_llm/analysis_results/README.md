# LLM Analysis Results

## Overview

This directory contains the results of LLM-based analysis on sampled issues from three LLM serving frameworks:

1. **vLLM**: 75 analysis groups
2. **SGLang**: 75 analysis groups
3. **llama.cpp**: 98 analysis groups

## Analysis Methods

Each framework has been analyzed using 10 different sampling methods:
- Label-based sampling (one group per label)
- Temporal sampling (time periods)
- Resolution time sampling (speed of closure)
- Complexity sampling (discussion levels)
- Author-based sampling (contributor types)
- Reaction-based sampling (community engagement)
- Long-tail sampling (label frequency)
- Cross-reference sampling (issue relationships)
- State transition sampling (lifecycle patterns)
- Anomaly sampling (outliers)

## Directory Structure

```
analysis_results/
├── vllm/
│   ├── [sampling_method]/
│   │   └── [sample_group]/
│   │       ├── RESEARCH_QUESTIONS.md
│   │       ├── analysis_results.json
│   │       ├── patterns.json
│   │       ├── findings.md
│   │       └── recommendations.md
│   └── cross_cutting/
├── sglang/
│   └── [same structure]
└── llama_cpp/
    └── [same structure]
```

## Research Questions

Each analysis group addresses specific research questions relevant to its sampling method.
See individual `RESEARCH_QUESTIONS.md` files for details.

## Status

⏳ All analyses are currently pending.

---
Generated: 2025-07-23 11:26:32
