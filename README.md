# Production Bug Analysis for LLM Serving Frameworks

A systematic research study analyzing production bugs and performance issues in major open-source LLM serving frameworks: vLLM, llama.cpp, and SGLang.

## 📊 Project Overview

This project analyzes 12,349 GitHub issues across three leading LLM serving frameworks to identify common production failure modes, bug patterns, and performance bottlenecks. Our goal is to improve the reliability and robustness of LLM deployments in production environments.

### Key Statistics
- **vLLM**: 4,078 issues analyzed (2,225 bugs/performance issues)
- **llama.cpp**: 5,470 issues analyzed (2,601 bugs/performance issues)  
- **SGLang**: 2,567 issues analyzed (106 bugs/performance issues)

## 🎯 Research Objectives

1. Identify and categorize common production bugs in LLM serving systems
2. Create a comprehensive bug taxonomy for LLM infrastructure
3. Analyze critical failure modes and their root causes
4. Provide actionable recommendations for improving production reliability
5. Publish findings to help the community build more robust LLM systems

## 📁 Repository Structure

```
.
├── docs/
│   └── research_paper/      # Research documentation and findings
│       ├── research_plan.md
│       ├── bug_taxonomy.md
│       ├── critical_bugs_analysis.md
│       └── paper_outline.md
├── scripts/                 # Data collection and analysis tools
│   ├── scrape_github_issues.py
│   ├── analyze_issues.py
│   └── issues_datasets/     # JSON datasets of scraped issues
├── vllm/                    # vLLM framework demo and testing
├── llama-cpp-demo/          # llama.cpp framework demo
├── sglang-demo/             # SGLang framework demo
└── scheduler/               # Additional research components
```

## 🐛 Bug Taxonomy

Our analysis identified 7 major categories of production bugs:

1. **Memory Management Issues** - OOM errors, memory leaks, VRAM fragmentation
2. **Concurrency & Synchronization** - Race conditions, deadlocks, cache corruption
3. **GPU/CUDA Issues** - Device errors, multi-GPU synchronization problems
4. **API/Protocol Issues** - Request handling failures, streaming errors
5. **Model-Specific Bugs** - Loading failures, inference errors
6. **Performance Degradation** - Latency spikes, throughput bottlenecks
7. **Scaling & Distribution** - Multi-GPU coordination, cluster issues

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- (Optional) CUDA-capable GPU for framework testing

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yunwei37/vllm-exp.git
cd vllm-exp
```

2. Set up Python environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. (Optional) Set up framework demos:
```bash
# For vLLM demo
cd vllm && bash quick_start.sh

# For llama.cpp demo
cd llama-cpp-demo && bash quick_start.sh

# For SGLang demo
cd sglang-demo && bash quick_start.sh
```

## 🔧 Usage

### Analyze GitHub Issues

1. Scrape issues from GitHub (requires GitHub token):
```bash
python scripts/scrape_github_issues.py --repo vllm-project/vllm --output vllm_issues.json
```

2. Analyze scraped issues:
```bash
python scripts/analyze_issues.py --input vllm_issues.json --output analysis_results.json
```

### Explore Research Findings

See our comprehensive research documentation:
- [Research Plan](docs/research_paper/research_plan.md) - Overview of methodology
- [Bug Taxonomy](docs/research_paper/bug_taxonomy.md) - Detailed bug categorization
- [Critical Bugs Analysis](docs/research_paper/critical_bugs_analysis.md) - Deep dive into severe issues

## 📈 Key Findings

1. **Memory management** is the #1 cause of production failures across all frameworks
2. **Concurrency bugs** are particularly challenging due to non-deterministic behavior
3. **GPU/CUDA issues** often manifest only under high load or specific hardware
4. **API compatibility** problems frequently occur during framework upgrades
5. **Performance regressions** are common but often go unnoticed until production

## 🤝 Contributing

We welcome contributions! Areas where help is needed:
- Expanding the bug taxonomy with new categories
- Adding analysis for more LLM frameworks
- Improving issue classification accuracy
- Creating automated bug detection tools

## 📝 Citation

If you use this research in your work, please cite:
```
@misc{llm-production-bugs-2025,
  title={A Systematic Study of Production Bugs in LLM Serving Frameworks},
  author={[Authors]},
  year={2025},
  url={https://github.com/yunwei37/vllm-exp}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The vLLM, llama.cpp, and SGLang communities for their open-source contributions
- All issue reporters who helped identify and document these bugs
- Researchers and engineers working to improve LLM serving reliability