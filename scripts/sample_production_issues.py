#!/usr/bin/env python3
"""
Sample and analyze production bugs and performance issues from LLM serving frameworks
"""

import json
import os
from datetime import datetime
from collections import defaultdict
import re

def load_issues(filename):
    """Load issues from JSON file"""
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    if 'items' in data:
        return data['items']
    elif 'issues' in data:
        return data['issues']
    return None

def is_production_issue(issue):
    """Check if issue is related to production deployment"""
    title = issue.get('title', '').lower()
    body = issue.get('body', '').lower() if issue.get('body') else ''
    
    production_keywords = [
        'production', 'deploy', 'server', 'api', 'endpoint', 'request',
        'timeout', 'latency', 'throughput', 'qps', 'memory leak', 'oom',
        'cuda', 'gpu', 'vram', 'inference', 'serving', 'concurrent',
        'scaling', 'load', 'performance', 'slow', 'hang', 'crash',
        'memory', 'leak', 'error', 'fail', 'exception'
    ]
    
    text = title + ' ' + body
    return any(keyword in text for keyword in production_keywords)

def categorize_issue(issue):
    """Categorize issue by type"""
    title = issue.get('title', '').lower()
    body = issue.get('body', '').lower() if issue.get('body') else ''
    text = title + ' ' + body
    
    categories = {
        'memory': ['oom', 'memory leak', 'vram', 'gpu memory', 'out of memory', 'memory error'],
        'performance': ['slow', 'latency', 'throughput', 'performance', 'speed', 'qps', 'inference time'],
        'concurrency': ['concurrent', 'parallel', 'multi', 'thread', 'async', 'deadlock', 'race condition'],
        'gpu': ['cuda', 'gpu', 'nccl', 'cudnn', 'tensor', 'device'],
        'api': ['api', 'endpoint', 'request', 'response', 'timeout', 'http', 'rest'],
        'crash': ['crash', 'seg fault', 'core dump', 'hang', 'freeze', 'unresponsive'],
        'model': ['model', 'weight', 'loading', 'quantization', 'precision', 'fp16', 'int8'],
        'scaling': ['scale', 'distributed', 'cluster', 'multi-node', 'load balance']
    }
    
    issue_categories = []
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            issue_categories.append(category)
    
    return issue_categories if issue_categories else ['other']

def extract_error_info(issue):
    """Extract error messages and stack traces"""
    body = issue.get('body', '') if issue.get('body') else ''
    
    # Look for error messages
    error_patterns = [
        r'error:?\s*([^\n]+)',
        r'exception:?\s*([^\n]+)',
        r'traceback[\s\S]*?(?=\n\n|\Z)',
        r'cuda.*error[^\n]+',
        r'assert.*failed[^\n]+'
    ]
    
    errors = []
    for pattern in error_patterns:
        matches = re.findall(pattern, body, re.IGNORECASE)
        errors.extend(matches[:2])  # Limit to first 2 matches per pattern
    
    return errors

def analyze_repository_issues(name, issues):
    """Analyze production issues for a repository"""
    if not issues:
        return None
    
    # Filter for production-related issues
    production_issues = [i for i in issues if is_production_issue(i)]
    
    # Filter for bug/performance labels
    bug_issues = []
    for issue in production_issues:
        labels = [label['name'].lower() for label in issue.get('labels', [])]
        if any(l in labels for l in ['bug', 'performance', 'crash', 'error', 'bug-unconfirmed']):
            bug_issues.append(issue)
    
    # Categorize issues
    category_counts = defaultdict(int)
    sample_issues = defaultdict(list)
    
    for issue in bug_issues[:500]:  # Analyze first 500 bug issues
        categories = categorize_issue(issue)
        for cat in categories:
            category_counts[cat] += 1
            if len(sample_issues[cat]) < 5:  # Keep 5 samples per category
                sample_issues[cat].append({
                    'number': issue['number'],
                    'title': issue['title'],
                    'state': issue['state'],
                    'created_at': issue['created_at'],
                    'labels': [l['name'] for l in issue.get('labels', [])],
                    'errors': extract_error_info(issue)[:2]  # First 2 errors
                })
    
    return {
        'name': name,
        'total_production_issues': len(production_issues),
        'bug_issues': len(bug_issues),
        'categories': dict(category_counts),
        'samples': dict(sample_issues)
    }

def main():
    print("Analyzing Production Bugs and Performance Issues in LLM Serving Frameworks")
    print("=" * 80)
    
    # Create output directory
    os.makedirs('analysis_output', exist_ok=True)
    
    datasets = [
        ('vLLM', 'vllm_all_issues.json'),
        ('llama.cpp', 'llama_cpp_issues.json'),
        ('SGLang', 'sglang_issues.json')
    ]
    
    all_results = {}
    
    for name, filename in datasets:
        print(f"\nAnalyzing {name}...")
        issues = load_issues(filename)
        if issues:
            result = analyze_repository_issues(name, issues)
            if result:
                all_results[name] = result
                
                # Print summary
                print(f"  Production-related issues: {result['total_production_issues']}")
                print(f"  Bug/Performance issues: {result['bug_issues']}")
                print(f"  Top categories:")
                for cat, count in sorted(result['categories'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    - {cat}: {count}")
                
                # Save detailed results
                with open(f"analysis_output/{name.lower()}_production_issues.json", 'w') as f:
                    json.dump(result, f, indent=2)
    
    # Create summary report
    print("\n" + "=" * 80)
    print("SUMMARY OF PRODUCTION ISSUES ACROSS FRAMEWORKS")
    print("=" * 80)
    
    # Common issue patterns
    all_categories = defaultdict(lambda: {'count': 0, 'frameworks': []})
    
    for framework, result in all_results.items():
        for cat, count in result['categories'].items():
            all_categories[cat]['count'] += count
            all_categories[cat]['frameworks'].append(framework)
    
    print("\nMost Common Issue Categories:")
    for cat, data in sorted(all_categories.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
        frameworks = ', '.join(data['frameworks'])
        print(f"  {cat}: {data['count']} issues across {frameworks}")
    
    # Save combined analysis
    with open('analysis_output/combined_analysis.json', 'w') as f:
        json.dump({
            'frameworks': all_results,
            'common_categories': dict(all_categories),
            'analysis_date': datetime.now().isoformat()
        }, f, indent=2)
    
    print("\nDetailed analysis saved to analysis_output/")

if __name__ == "__main__":
    main()