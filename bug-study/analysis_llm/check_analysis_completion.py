#!/usr/bin/env python3
"""
Check analysis completion rate by comparing findings.md file sizes.
Generate a todo list of unfinished analyses.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict
import json


def check_completion_status(base_dir: str, min_size_threshold: int = 1000) -> Dict:
    """
    Check completion status of analyses based on file size.
    
    Args:
        base_dir: Base directory to check
        min_size_threshold: Minimum file size in bytes to consider as completed
        
    Returns:
        Dictionary with completion statistics and lists
    """
    base_path = Path(base_dir)
    
    completed = []
    unfinished = []
    all_analyses = []
    
    # Find all findings.md files
    findings_files = list(base_path.rglob("findings.md"))
    
    for file_path in findings_files:
        file_size = file_path.stat().st_size
        relative_path = file_path.relative_to(base_path)
        
        # Extract analysis info from path
        parts = relative_path.parts[:-1]  # Remove 'findings.md'
        analysis_name = "/".join(parts)
        
        analysis_info = {
            'path': str(file_path),
            'relative_path': str(relative_path),
            'analysis_name': analysis_name,
            'size': file_size,
            'completed': file_size >= min_size_threshold
        }
        
        all_analyses.append(analysis_info)
        
        if file_size >= min_size_threshold:
            completed.append(analysis_info)
        else:
            unfinished.append(analysis_info)
    
    # Sort by size for better overview
    completed.sort(key=lambda x: x['size'], reverse=True)
    unfinished.sort(key=lambda x: x['analysis_name'])
    
    return {
        'total': len(all_analyses),
        'completed': completed,
        'unfinished': unfinished,
        'completion_rate': len(completed) / len(all_analyses) * 100 if all_analyses else 0
    }


def generate_todo_list(unfinished: List[Dict]) -> List[str]:
    """Generate a prioritized todo list from unfinished analyses."""
    todo_list = []
    
    # Group by analysis type
    grouped = {}
    for item in unfinished:
        parts = item['analysis_name'].split('/')
        if len(parts) > 0:
            framework = parts[0]
            if framework not in grouped:
                grouped[framework] = []
            grouped[framework].append(item)
    
    # Prioritize by framework and type
    priority_order = ['vllm', 'sglang', 'llama_cpp']
    
    for framework in priority_order:
        if framework in grouped:
            # Further prioritize by sampling method
            label_based = []
            other_methods = []
            
            for item in grouped[framework]:
                if 'label_based' in item['analysis_name']:
                    label_based.append(item)
                else:
                    other_methods.append(item)
            
            # Add label-based first
            for item in sorted(label_based, key=lambda x: x['analysis_name']):
                todo_list.append(item['analysis_name'])
            
            # Then other methods
            for item in sorted(other_methods, key=lambda x: x['analysis_name']):
                todo_list.append(item['analysis_name'])
    
    return todo_list


def print_completion_report(results: Dict):
    """Print a formatted completion report."""
    print("=" * 80)
    print("ANALYSIS COMPLETION REPORT")
    print("=" * 80)
    print(f"\nTotal analyses found: {results['total']}")
    print(f"Completed: {len(results['completed'])} ({results['completion_rate']:.1f}%)")
    print(f"Unfinished: {len(results['unfinished'])} ({100 - results['completion_rate']:.1f}%)")
    
    print("\n" + "-" * 40)
    print("COMPLETED ANALYSES (by size):")
    print("-" * 40)
    for item in results['completed']:
        print(f"✓ {item['analysis_name']:<60} {item['size']:>10,} bytes")
    
    print("\n" + "-" * 40)
    print("UNFINISHED ANALYSES:")
    print("-" * 40)
    for item in results['unfinished']:
        print(f"✗ {item['analysis_name']:<60} {item['size']:>10} bytes")
    
    # Generate and print todo list
    todo_list = generate_todo_list(results['unfinished'])
    
    print("\n" + "=" * 80)
    print("TODO LIST (Prioritized):")
    print("=" * 80)
    for i, task in enumerate(todo_list, 1):
        print(f"{i:3d}. {task}")
    
    # Group by framework for summary
    framework_summary = {}
    for item in results['unfinished']:
        framework = item['analysis_name'].split('/')[0]
        if framework not in framework_summary:
            framework_summary[framework] = 0
        framework_summary[framework] += 1
    
    print("\n" + "-" * 40)
    print("SUMMARY BY FRAMEWORK:")
    print("-" * 40)
    for framework, count in sorted(framework_summary.items()):
        print(f"{framework}: {count} unfinished analyses")

def main():
    """Main function to check completion status."""
    # Check analysis_results directory
    analysis_dir = "/root/yunwei37/vllm-exp/bug-study/analysis_llm/analysis_results"
    
    if not os.path.exists(analysis_dir):
        print(f"Error: Directory {analysis_dir} not found!")
        return
    
    # Check completion status
    results = check_completion_status(analysis_dir)
    
    # Print report
    print_completion_report(results)
    
    # Also check if sample_results has corresponding markdown files
    print("\n" + "=" * 80)
    print("CHECKING SAMPLE AVAILABILITY:")
    print("=" * 80)
    
    sample_dir = "/root/yunwei37/vllm-exp/bug-study/analysis_llm/markdown_samples"
    
    for item in results['unfinished'][:10]:  # Check first 10 unfinished
        # Construct expected markdown path
        parts = item['analysis_name'].split('/')
        expected_md = Path(sample_dir) / Path(*parts) / "issues.md"
        
        if expected_md.exists():
            size = expected_md.stat().st_size
            print(f"✓ Samples available for {item['analysis_name']} ({size:,} bytes)")
        else:
            print(f"✗ No samples found for {item['analysis_name']}")


if __name__ == "__main__":
    main()