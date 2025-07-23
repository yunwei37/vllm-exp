#!/usr/bin/env python3
"""Merge vllm_closed_issues_all.json and vllm_open_issues_all.json into vllm_all_issues.json"""

import json
import os

def merge_issue_files():
    base_dir = "/root/yunwei37/vllm-exp/bug-study/data"
    
    # File paths
    closed_file = os.path.join(base_dir, "vllm_closed_issues_all.json")
    open_file = os.path.join(base_dir, "vllm_open_issues_all.json")
    output_file = os.path.join(base_dir, "vllm_all_issues.json")
    
    # Read closed issues
    print(f"Reading closed issues from {closed_file}...")
    with open(closed_file, 'r') as f:
        closed_issues = json.load(f)
    
    # Read open issues
    print(f"Reading open issues from {open_file}...")
    with open(open_file, 'r') as f:
        open_issues = json.load(f)
    
    # Merge the issues
    if isinstance(closed_issues, list) and isinstance(open_issues, list):
        all_issues = closed_issues + open_issues
    elif isinstance(closed_issues, dict) and isinstance(open_issues, dict):
        all_issues = {**closed_issues, **open_issues}
    else:
        raise ValueError("Both files must contain either lists or dictionaries")
    
    # Write merged data
    print(f"Writing merged data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_issues, f, indent=2)
    
    # Print statistics
    if isinstance(all_issues, list):
        print(f"Total issues merged: {len(all_issues)}")
        print(f"- Closed issues: {len(closed_issues)}")
        print(f"- Open issues: {len(open_issues)}")
    else:
        print(f"Total keys merged: {len(all_issues)}")
        print(f"- From closed issues: {len(closed_issues)}")
        print(f"- From open issues: {len(open_issues)}")
    
    print(f"Successfully merged to {output_file}")

if __name__ == "__main__":
    merge_issue_files()