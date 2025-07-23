#!/usr/bin/env python3
"""
Helper script to run all analyses for a single framework
"""

import sys
import importlib.util
from pathlib import Path

def run_framework_analysis(framework_name, data_path):
    """Run all analyses for a single framework"""
    print(f"\n{'='*60}")
    print(f"Running complete analysis for {framework_name}")
    print(f"Data: {data_path}")
    print(f"{'='*60}\n")
    
    base_dir = Path(__file__).parent
    results_dir = base_dir / 'results'
    
    # Analysis modules to run
    analyses = [
        ('01_temporal_analysis.py', 'TemporalAnalyzer', 'temporal'),
        ('02_user_behavior_analysis.py', 'UserBehaviorAnalyzer', 'user_behavior'),
        ('03_label_complexity_analysis.py', 'LabelComplexityAnalyzer', 'label_complexity'),
        ('04_state_transition_analysis.py', 'StateTransitionAnalyzer', 'state_transition')
    ]
    
    for script_name, class_name, output_subdir in analyses:
        print(f"\nRunning {script_name}...")
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location("module", base_dir / script_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get analyzer class
        analyzer_class = getattr(module, class_name)
        
        # Run analysis
        output_dir = results_dir / output_subdir
        analyzer = analyzer_class(framework_name, data_path, output_dir)
        analyzer.run()
        
        print(f"✓ {script_name} complete")
    
    print(f"\n{'='*60}")
    print(f"All analyses complete for {framework_name}")
    print(f"Results saved in: {results_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_single_framework.py <framework_name> <data_path>")
        print("Example: python run_single_framework.py vllm ../data/vllm_all_issues.json")
        sys.exit(1)
    
    framework_name = sys.argv[1]
    data_path = sys.argv[2]
    
    run_framework_analysis(framework_name, data_path)