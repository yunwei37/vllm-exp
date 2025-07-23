# LLM Serving Framework Bug Study

A comprehensive research project analyzing production bugs and performance issues in modern LLM serving frameworks (vLLM, llama.cpp, and SGLang).

## Project Structure

```
bug-study/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── scraper/                  # Data collection scripts
│   ├── github_issues_scraper.py     # Main issue scraper
│   ├── github_issues_scraper_v2.py  # Enhanced scraper with filtering
│   ├── test_scraper.py              # Scraper testing utilities
│   └── sample_production_issues.py  # Sample data generator
├── data/                     # Collected and processed data
│   ├── vllm_all_issues.json         # All vLLM issues
│   ├── vllm_open_issues_all.json    # Open vLLM issues
│   ├── vllm_closed_issues_all.json  # Closed vLLM issues
│   ├── llama_cpp_issues.json        # llama.cpp issues
│   ├── sglang_issues.json           # SGLang issues
│   ├── vllm_production_issues.json  # Filtered production issues
│   ├── llama.cpp_production_issues.json
│   ├── sglang_production_issues.json
│   └── combined_analysis.json       # Combined analysis results
├── analysis/                 # Analysis scripts and tools
│   ├── analyze_issues.py     # Main analysis script
│   └── example_usage.py      # Example analysis workflows
└── doc/                      # Documentation and research plans
    ├── PAPER_OUTLINE.md      # Research paper outline
    ├── RESEARCH_PLAN.md      # Practical research methodology
    └── v1/                   # Version 1 documentation
        ├── BUG_TAXONOMY.md
        ├── CRITICAL_BUGS_ANALYSIS.md
        └── analysis_output/

```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Set up GitHub token (optional but recommended for higher rate limits)
export GITHUB_TOKEN="your_github_token"
```

### 2. Collect Data

```bash
# Scrape issues from repositories
cd scraper
python github_issues_scraper_v2.py

# Test with a smaller sample
python test_scraper.py
```

### 3. Analyze Data

```bash
# Run main analysis
cd ../analysis
python analyze_issues.py

# See example workflows
python example_usage.py
```

## Research Questions

1. **What types of bugs most frequently disrupt LLM production services?**
2. **What are the common symptoms that precede critical failures?**
3. **Which bugs can be detected with simple runtime checks?**
4. **What minimal information is needed to reproduce bugs effectively?**
5. **How do bug patterns differ across deployment scales?**

## Data Collection

The `scraper/` directory contains tools for collecting GitHub issues:

- **github_issues_scraper.py**: Basic scraper for fetching all issues
- **github_issues_scraper_v2.py**: Enhanced version with production filtering
- **test_scraper.py**: Testing utilities for validating scraper functionality
- **sample_production_issues.py**: Generates sample data for testing

### Supported Repositories

- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- [sgl-project/sglang](https://github.com/sgl-project/sglang)

### Scraper Features

- 🚦 **Smart Rate Limiting**: Automatically handles GitHub API rate limits
- 🔐 **Authentication Support**: Works with or without GitHub tokens
- 🎯 **Production Filtering**: Identifies production-relevant issues
- 📊 **Progress Tracking**: Real-time updates and summaries
- 💾 **JSON Export**: Structured data format
- 🔄 **Resume Support**: Checkpoint-based resumption

## Analysis Pipeline

The `analysis/` directory contains scripts for analyzing collected data:

1. **Categorization**: LLM-assisted classification of bugs
2. **Pattern Extraction**: Identifying common failure modes
3. **Root Cause Analysis**: Understanding underlying issues
4. **Tool Development**: Creating detection and prevention tools

## Data Files

The `data/` directory contains:

- **Raw Issue Data**: Complete issue exports from each framework
- **Filtered Data**: Production-relevant issues only
- **Analysis Results**: Processed insights and patterns

## Key Findings

See `doc/v1/` for detailed findings:

- **BUG_TAXONOMY.md**: Comprehensive bug classification
- **CRITICAL_BUGS_ANALYSIS.md**: Analysis of high-impact issues

## Usage Examples

### Scraping Issues

```bash
cd scraper

# Scrape all vLLM issues
python github_issues_scraper.py vllm-project/vllm --output ../data/vllm_all_issues.json

# Scrape with production filtering
python github_issues_scraper_v2.py

# Test scraper on small sample
python test_scraper.py
```

### Analyzing Data

```bash
cd analysis

# Basic analysis
python analyze_issues.py --input ../data/vllm_all_issues.json

# Compare frameworks
python analyze_issues.py --compare vllm llama.cpp sglang

# Generate bug detection rules
python analyze_issues.py --generate-detectors
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your analysis or improvements
4. Submit a pull request

## Future Work

- [ ] Automated bug detection tools
- [ ] Real-time monitoring integration
- [ ] Expanded framework coverage
- [ ] Machine learning-based bug prediction
- [ ] Interactive dashboard for results

## Research Outputs

- Research paper (in progress)
- Bug detection tool library
- Best practices guide
- Issue reporting templates

## License

This research is conducted for academic purposes. Please cite appropriately if using this work.

## Contact

For questions or collaborations, please open an issue in this repository.