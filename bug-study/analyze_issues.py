#!/usr/bin/env python3
"""
Analyze and compare issues from vLLM, llama.cpp, and SGLang repositories
"""

import json
import os
from datetime import datetime
from collections import Counter
import pandas as pd

def load_issues(filename):
    """Load issues from JSON file"""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        return None
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if 'items' in data:
        return data['items'], data.get('metadata', {})
    elif 'issues' in data:
        return data['issues'], data.get('metadata', {})
    else:
        print(f"Warning: Unexpected structure in {filename}")
        return None, None

def analyze_repository(name, issues):
    """Analyze issues for a single repository"""
    if not issues:
        return None
    
    # Separate issues and PRs
    real_issues = [i for i in issues if 'pull_request' not in i]
    
    # Basic stats
    total = len(real_issues)
    open_count = sum(1 for i in real_issues if i['state'] == 'open')
    closed_count = total - open_count
    
    # Label analysis
    label_counter = Counter()
    for issue in real_issues:
        for label in issue.get('labels', []):
            label_counter[label['name']] += 1
    
    # Date analysis (for recent activity)
    recent_issues = 0
    for issue in real_issues:
        created_date = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
        days_old = (datetime.now(created_date.tzinfo) - created_date).days
        if days_old <= 30:
            recent_issues += 1
    
    return {
        'name': name,
        'total_issues': total,
        'open_issues': open_count,
        'closed_issues': closed_count,
        'open_rate': f"{(open_count/total)*100:.1f}%" if total > 0 else "0%",
        'recent_30_days': recent_issues,
        'top_labels': label_counter.most_common(10)
    }

def main():
    print("=" * 80)
    print("LLM Serving Frameworks - Issue Analysis")
    print("=" * 80)
    
    # Load all issues
    datasets = [
        ('vLLM', 'vllm_all_issues.json'),
        ('llama.cpp', 'llama_cpp_issues.json'),
        ('SGLang', 'sglang_issues.json')
    ]
    
    results = []
    
    for name, filename in datasets:
        issues, metadata = load_issues(filename)
        if issues:
            analysis = analyze_repository(name, issues)
            if analysis:
                results.append(analysis)
                
                print(f"\n{name} Analysis:")
                print(f"  Total Issues: {analysis['total_issues']:,}")
                print(f"  Open: {analysis['open_issues']:,} ({analysis['open_rate']})")
                print(f"  Closed: {analysis['closed_issues']:,}")
                print(f"  Recent (30 days): {analysis['recent_30_days']}")
                print(f"  Top Labels:")
                for label, count in analysis['top_labels'][:5]:
                    print(f"    - {label}: {count}")
    
    # Comparative analysis
    print("\n" + "=" * 80)
    print("Comparative Analysis")
    print("=" * 80)
    
    if len(results) == 3:
        # Create comparison table
        print("\n📊 Overview:")
        print(f"{'Project':<15} {'Total Issues':<12} {'Open':<10} {'Closed':<10} {'Open Rate':<10} {'Recent':<10}")
        print("-" * 70)
        for r in results:
            print(f"{r['name']:<15} {r['total_issues']:<12,} {r['open_issues']:<10,} {r['closed_issues']:<10,} {r['open_rate']:<10} {r['recent_30_days']:<10}")
        
        # Health indicators
        print("\n📈 Project Health Indicators:")
        for r in results:
            health_score = 0
            issues = []
            
            # Open rate between 10-30% is healthy
            open_rate = float(r['open_rate'].rstrip('%'))
            if 10 <= open_rate <= 30:
                health_score += 1
            else:
                issues.append(f"open rate {r['open_rate']}")
            
            # Recent activity
            if r['recent_30_days'] > 20:
                health_score += 1
            else:
                issues.append(f"low recent activity ({r['recent_30_days']} in 30 days)")
            
            # Issue management (check for stale labels)
            stale_count = next((count for label, count in r['top_labels'] if 'stale' in label.lower() or 'inactive' in label.lower()), 0)
            if r['total_issues'] > 0:
                stale_ratio = stale_count / r['total_issues']
                if stale_ratio < 0.3:
                    health_score += 1
                else:
                    issues.append(f"high stale ratio ({stale_ratio:.1%})")
            
            print(f"\n{r['name']}:")
            print(f"  Health Score: {health_score}/3")
            if issues:
                print(f"  Concerns: {', '.join(issues)}")
            else:
                print(f"  Status: Healthy issue management")
        
        # Common issue patterns
        print("\n🔍 Common Patterns:")
        
        # Collect all labels
        all_labels = {}
        for r in results:
            all_labels[r['name']] = dict(r['top_labels'])
        
        # Find common labels
        common_labels = ['bug', 'enhancement', 'feature', 'documentation', 'help wanted', 'good first issue']
        
        print(f"\n{'Label':<20} {'vLLM':<10} {'llama.cpp':<10} {'SGLang':<10}")
        print("-" * 50)
        for label in common_labels:
            counts = []
            for r in results:
                count = next((c for l, c in r['top_labels'] if label in l.lower()), 0)
                counts.append(str(count) if count > 0 else '-')
            print(f"{label:<20} {counts[0]:<10} {counts[1]:<10} {counts[2]:<10}")

if __name__ == "__main__":
    main()