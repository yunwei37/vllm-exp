#!/usr/bin/env python3
"""
Runner script for individual analyses
"""

import sys
import importlib.util

def run_analysis(module_file, class_name, framework, data_path, output_dir):
    """Run a specific analysis module"""
    # Load the module
    spec = importlib.util.spec_from_file_location("analysis_module", module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the analyzer class
    analyzer_class = getattr(module, class_name)
    
    # Run the analysis
    analyzer = analyzer_class(framework, data_path, output_dir)
    analyzer.run()
    
    print(f"✓ {framework} analysis complete")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: run_analysis.py <module_file> <class_name> <framework> <data_path> <output_dir>")
        sys.exit(1)
    
    module_file = sys.argv[1]
    class_name = sys.argv[2]
    framework = sys.argv[3]
    data_path = sys.argv[4]
    output_dir = sys.argv[5]
    
    run_analysis(module_file, class_name, framework, data_path, output_dir)