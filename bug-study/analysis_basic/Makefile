# Makefile for LLM Framework Issue Analysis
# Run comprehensive data mining analysis on vLLM, SGLang, and llama.cpp issues

# Configuration
PYTHON = python3
DATA_DIR = ../data
RESULTS_DIR = results
SCRIPTS = 01_temporal_analysis.py \
          02_user_behavior_analysis.py \
          03_label_complexity_analysis.py \
          04_state_transition_analysis.py \
          05_cross_framework_comparison.py

# Framework data files
VLLM_DATA = $(DATA_DIR)/vllm_all_issues.json
SGLANG_DATA = $(DATA_DIR)/sglang_issues.json
LLAMA_CPP_DATA = $(DATA_DIR)/llama_cpp_issues.json

# Ensure vLLM data is merged before analysis
$(VLLM_DATA): $(DATA_DIR)/vllm_open_issues_all.json $(DATA_DIR)/vllm_closed_issues_all.json
	@echo "Merging vLLM issues..."
	@cd $(DATA_DIR) && $(PYTHON) merge_vllm_issues_fixed.py

# Default target: run all analyses
.PHONY: all
all: prepare temporal user label state cross report

# Prepare directories
.PHONY: prepare
prepare:
	@echo "Preparing directories..."
	@mkdir -p $(RESULTS_DIR)/{temporal,user_behavior,label_complexity,state_transition,cross_framework,plots}

# Individual analysis targets
.PHONY: temporal
temporal: $(VLLM_DATA)
	@echo "\n=== Running Temporal Analysis ==="
	@echo "Analyzing vLLM..."
	@$(PYTHON) 01_temporal_analysis.py || true
	@echo "Analyzing SGLang..."
	@$(PYTHON) run_analysis.py 01_temporal_analysis.py TemporalAnalyzer sglang $(SGLANG_DATA) $(RESULTS_DIR)/temporal
	@echo "Analyzing llama.cpp..."
	@$(PYTHON) run_analysis.py 01_temporal_analysis.py TemporalAnalyzer llama_cpp $(LLAMA_CPP_DATA) $(RESULTS_DIR)/temporal

.PHONY: user
user: $(VLLM_DATA)
	@echo "\n=== Running User Behavior Analysis ==="
	@echo "Analyzing vLLM..."
	@$(PYTHON) 02_user_behavior_analysis.py || true
	@echo "Analyzing SGLang..."
	@$(PYTHON) run_analysis.py 02_user_behavior_analysis.py UserBehaviorAnalyzer sglang $(SGLANG_DATA) $(RESULTS_DIR)/user_behavior
	@echo "Analyzing llama.cpp..."
	@$(PYTHON) run_analysis.py 02_user_behavior_analysis.py UserBehaviorAnalyzer llama_cpp $(LLAMA_CPP_DATA) $(RESULTS_DIR)/user_behavior

.PHONY: label
label: $(VLLM_DATA)
	@echo "\n=== Running Label and Complexity Analysis ==="
	@echo "Analyzing vLLM..."
	@$(PYTHON) 03_label_complexity_analysis.py || true
	@echo "Analyzing SGLang..."
	@$(PYTHON) run_analysis.py 03_label_complexity_analysis.py LabelComplexityAnalyzer sglang $(SGLANG_DATA) $(RESULTS_DIR)/label_complexity
	@echo "Analyzing llama.cpp..."
	@$(PYTHON) run_analysis.py 03_label_complexity_analysis.py LabelComplexityAnalyzer llama_cpp $(LLAMA_CPP_DATA) $(RESULTS_DIR)/label_complexity

.PHONY: state
state: $(VLLM_DATA)
	@echo "\n=== Running State Transition Analysis ==="
	@echo "Analyzing vLLM..."
	@$(PYTHON) 04_state_transition_analysis.py || true
	@echo "Analyzing SGLang..."
	@$(PYTHON) run_analysis.py 04_state_transition_analysis.py StateTransitionAnalyzer sglang $(SGLANG_DATA) $(RESULTS_DIR)/state_transition
	@echo "Analyzing llama.cpp..."
	@$(PYTHON) run_analysis.py 04_state_transition_analysis.py StateTransitionAnalyzer llama_cpp $(LLAMA_CPP_DATA) $(RESULTS_DIR)/state_transition

.PHONY: cross
cross: $(VLLM_DATA)
	@echo "\n=== Running Cross-Framework Comparison ==="
	@$(PYTHON) 05_cross_framework_comparison.py

# Generate comprehensive report
.PHONY: report
report:
	@echo "\n=== Generating Master Report ==="
	@$(PYTHON) generate_master_report.py

# Run basic analysis (from original script)
.PHONY: basic
basic: $(VLLM_DATA)
	@echo "\n=== Running Basic Analysis ==="
	@$(PYTHON) basic_analysis.py

# Clean up results
.PHONY: clean
clean:
	@echo "Cleaning results directory..."
	@rm -rf $(RESULTS_DIR)

# Run specific framework analysis
.PHONY: vllm
vllm: $(VLLM_DATA)
	@echo "\n=== Running all analyses for vLLM ==="
	@$(PYTHON) run_single_framework.py vllm $(VLLM_DATA)

.PHONY: sglang
sglang:
	@echo "\n=== Running all analyses for SGLang ==="
	@$(PYTHON) run_single_framework.py sglang $(SGLANG_DATA)

.PHONY: llama
llama:
	@echo "\n=== Running all analyses for llama.cpp ==="
	@$(PYTHON) run_single_framework.py llama_cpp $(LLAMA_CPP_DATA)

# Quick analysis (temporal + basic metrics only)
.PHONY: quick
quick: prepare temporal cross
	@echo "\n=== Quick analysis complete ==="

# Help
.PHONY: help
help:
	@echo "LLM Framework Issue Analysis"
	@echo "============================"
	@echo "Available targets:"
	@echo "  all      - Run all analyses (default)"
	@echo "  temporal - Run temporal analysis"
	@echo "  user     - Run user behavior analysis"
	@echo "  label    - Run label and complexity analysis"
	@echo "  state    - Run state transition analysis"
	@echo "  cross    - Run cross-framework comparison"
	@echo "  report   - Generate master report"
	@echo "  basic    - Run basic analysis"
	@echo "  vllm     - Run all analyses for vLLM only"
	@echo "  sglang   - Run all analyses for SGLang only"
	@echo "  llama    - Run all analyses for llama.cpp only"
	@echo "  quick    - Run quick analysis (temporal + cross)"
	@echo "  clean    - Remove all results"
	@echo "  help     - Show this help message"

# Check dependencies
.PHONY: check-deps
check-deps:
	@echo "Checking Python dependencies..."
	@$(PYTHON) -c "import pandas, numpy, matplotlib, seaborn, networkx" || \
		(echo "Missing dependencies. Please install: pip install pandas numpy matplotlib seaborn networkx" && exit 1)

# Install dependencies
.PHONY: install-deps
install-deps:
	@echo "Installing Python dependencies..."
	@pip install pandas numpy matplotlib seaborn networkx