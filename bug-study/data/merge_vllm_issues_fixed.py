#!/usr/bin/env python3
"""Merge vllm_closed_issues_all.json and vllm_open_issues_all.json into vllm_all_issues.json"""

import json
import os
from datetime import datetime

def merge_issue_files():
    base_dir = "/root/yunwei37/vllm-exp/bug-study/data"
    
    # File paths
    closed_file = os.path.join(base_dir, "vllm_closed_issues_all.json")
    open_file = os.path.join(base_dir, "vllm_open_issues_all.json")
    output_file = os.path.join(base_dir, "vllm_all_issues.json")
    
    # Read closed issues
    print(f"Reading closed issues from {closed_file}...")
    with open(closed_file, 'r') as f:
        closed_data = json.load(f)
    
    # Read open issues
    print(f"Reading open issues from {open_file}...")
    with open(open_file, 'r') as f:
        open_data = json.load(f)
    
    # Extract items from both files
    closed_items = closed_data.get('items', [])
    open_items = open_data.get('items', [])
    
    print(f"Found {len(closed_items)} closed issues")
    print(f"Found {len(open_items)} open issues")
    
    # Merge the items
    all_items = closed_items + open_items
    
    # Create merged data structure with updated metadata
    merged_data = {
        'metadata': {
            'total_count': len(all_items),
            'closed_count': len(closed_items),
            'open_count': len(open_items),
            'merge_timestamp': datetime.now().isoformat(),
            'source_files': {
                'closed': closed_file,
                'open': open_file
            }
        },
        'items': all_items
    }
    
    # Write merged data
    print(f"Writing merged data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    # Print statistics
    print(f"\nMerge Summary:")
    print(f"Total issues merged: {len(all_items)}")
    print(f"- Closed issues: {len(closed_items)}")
    print(f"- Open issues: {len(open_items)}")
    print(f"Successfully merged to {output_file}")
    
    # Check file size
    file_size = os.path.getsize(output_file)
    print(f"\nOutput file size: {file_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    merge_issue_files()