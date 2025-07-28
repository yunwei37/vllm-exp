#!/usr/bin/env python3
"""
Script to extract individual issues from bulk JSON files and save them as separate JSON files.
"""

import json
import os
from pathlib import Path


def extract_issues(source_file, target_dir, issue_type):
    """
    Extract issues from a source JSON file and save each as an individual JSON file.
    
    Args:
        source_file: Path to the source issues.json file
        target_dir: Directory to save individual issue JSON files
        issue_type: Type of issues (bug or performance)
    """
    # Ensure target directory exists
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # Read source file
    try:
        with open(source_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Source file {source_file} not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {source_file}: {e}")
        return
    
    # Extract issues - the file is already a list of issues
    issues = data if isinstance(data, list) else data.get('issues', [])
    
    if not issues:
        print(f"No issues found in {source_file}")
        return
    
    print(f"Found {len(issues)} {issue_type} issues")
    
    # Save each issue as a separate JSON file
    for issue in issues:
        issue_number = issue.get('number')
        if not issue_number:
            print("Warning: Issue without number found, skipping")
            continue
        
        # Create filename
        filename = f"issue_{issue_number}.json"
        filepath = os.path.join(target_dir, filename)
        
        # Add metadata
        issue_data = {
            "issue_type": issue_type,
            "extracted_from": source_file,
            "issue": issue
        }
        
        # Save individual issue
        with open(filepath, 'w') as f:
            json.dump(issue_data, f, indent=2)
        
        print(f"  - Saved issue #{issue_number} to {filename}")
    
    print(f"Successfully extracted {len(issues)} {issue_type} issues to {target_dir}\n")


def main():
    """Main function to extract bug and performance issues."""
    
    base_dir = "/root/yunwei37/vllm-exp/bug-study/analysis_llm"
    
    # Define source files and target directories
    tasks = [
        {
            "source": os.path.join(base_dir, "sample_results/vllm/label_based/bug/issues.json"),
            "target": os.path.join(base_dir, "systematic_classification_framework/vllm_bugs"),
            "type": "bug"
        },
        {
            "source": os.path.join(base_dir, "sample_results/vllm/label_based/performance/issues.json"),
            "target": os.path.join(base_dir, "systematic_classification_framework/vllm_performance"),
            "type": "performance"
        }
    ]
    
    print("Starting issue extraction...\n")
    
    # Process each task
    for task in tasks:
        print(f"Processing {task['type']} issues:")
        extract_issues(task['source'], task['target'], task['type'])
    
    print("Issue extraction completed!")


if __name__ == "__main__":
    main()