#!/usr/bin/env python3
import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Any

def analyze_bug_issues(filepath: str) -> Dict[str, Any]:
    """Analyze bug issues from vLLM repository"""
    
    with open(filepath, 'r') as f:
        issues = json.load(f)
    
    results = {
        'total_issues': len(issues),
        'issue_types': defaultdict(list),
        'technical_problems': defaultdict(list),
        'resolution_patterns': defaultdict(list),
        'information_quality': {
            'missing_info': [],
            'well_documented': [],
            'has_environment_info': 0,
            'has_error_messages': 0,
            'has_reproduction_steps': 0
        },
        'state_distribution': Counter(),
        'label_distribution': Counter(),
        'common_keywords': Counter()
    }
    
    # Keywords for categorization
    error_keywords = {
        'KeyError': 'key_error',
        'RuntimeError': 'runtime_error',
        'TypeError': 'type_error',
        'AttributeError': 'attribute_error',
        'ValueError': 'value_error',
        'CUDA': 'cuda_related',
        'GPU': 'gpu_related',
        'memory': 'memory_related',
        'OOM': 'out_of_memory',
        'segfault': 'segmentation_fault',
        'crash': 'crash',
        'hang': 'hang_freeze',
        'timeout': 'timeout',
        'tokenizer': 'tokenizer_issue',
        'model': 'model_loading',
        'tensor': 'tensor_operation',
        'shape': 'shape_mismatch',
        'compatibility': 'compatibility',
        'version': 'version_related',
        'install': 'installation',
        'import': 'import_error',
        'performance': 'performance',
        'slow': 'performance',
        'API': 'api_related',
        'request': 'request_handling',
        'response': 'response_issue'
    }
    
    for issue in issues:
        # Track state
        results['state_distribution'][issue['state']] += 1
        
        # Track all labels
        for label in issue.get('labels', []):
            results['label_distribution'][label] += 1
        
        # Analyze body content
        body = issue.get('body', '').lower() if issue.get('body') else ''
        title = issue.get('title', '').lower()
        
        # Check for environment information
        if 'environment' in body or 'pytorch version' in body or 'vllm version' in body:
            results['information_quality']['has_environment_info'] += 1
        
        # Check for error messages
        if 'error:' in body or 'traceback' in body or 'exception' in body:
            results['information_quality']['has_error_messages'] += 1
        
        # Check for reproduction steps
        if 'reproduce' in body or 'steps:' in body or 'how to' in body:
            results['information_quality']['has_reproduction_steps'] += 1
        
        # Categorize technical problems
        for keyword, category in error_keywords.items():
            if keyword.lower() in body or keyword.lower() in title:
                results['technical_problems'][category].append({
                    'number': issue['number'],
                    'title': issue['title'],
                    'html_url': issue['html_url']
                })
        
        # Analyze resolution patterns
        if issue['state'] == 'closed':
            if issue.get('comments', 0) == 0:
                results['resolution_patterns']['closed_without_discussion'].append(issue['number'])
            elif issue.get('comments', 0) > 5:
                results['resolution_patterns']['extensive_discussion'].append(issue['number'])
            
            # Check if marked as stale
            if 'stale' in issue.get('labels', []):
                results['resolution_patterns']['closed_as_stale'].append(issue['number'])
        
        # Categorize issue types based on title/body patterns
        if re.search(r'(safetensor|format|load|model)', title + body, re.I):
            results['issue_types']['model_loading'].append(issue['number'])
        
        if re.search(r'(tokeniz|token|unk)', title + body, re.I):
            results['issue_types']['tokenization'].append(issue['number'])
        
        if re.search(r'(cuda|gpu|memory|oom)', title + body, re.I):
            results['issue_types']['gpu_memory'].append(issue['number'])
        
        if re.search(r'(api|request|response|server)', title + body, re.I):
            results['issue_types']['api_serving'].append(issue['number'])
        
        if re.search(r'(guided|json|decoding)', title + body, re.I):
            results['issue_types']['guided_decoding'].append(issue['number'])
        
        if re.search(r'(tensor|shape|view|stride)', title + body, re.I):
            results['issue_types']['tensor_operations'].append(issue['number'])
        
        # Identify well-documented issues
        if (results['information_quality']['has_environment_info'] and 
            results['information_quality']['has_error_messages'] and
            len(body) > 500):
            results['information_quality']['well_documented'].append(issue['number'])
        elif len(body) < 100:
            results['information_quality']['missing_info'].append(issue['number'])
    
    return results

# Run analysis
results = analyze_bug_issues('/root/yunwei37/vllm-exp/bug-study/analysis_llm/sample_results/vllm/label_based/bug/issues.json')

# Print detailed report
print("=== vLLM Bug Label Analysis Report ===\n")

print(f"Total Issues Analyzed: {results['total_issues']}")
print(f"\n1. STATE DISTRIBUTION:")
for state, count in results['state_distribution'].items():
    percentage = (count / results['total_issues']) * 100
    print(f"   - {state}: {count} ({percentage:.1f}%)")

print(f"\n2. LABEL DISTRIBUTION:")
for label, count in sorted(results['label_distribution'].items(), key=lambda x: x[1], reverse=True):
    percentage = (count / results['total_issues']) * 100
    print(f"   - {label}: {count} ({percentage:.1f}%)")

print(f"\n3. ISSUE TYPES (overlapping categories):")
for issue_type, issue_nums in results['issue_types'].items():
    if issue_nums:
        percentage = (len(issue_nums) / results['total_issues']) * 100
        print(f"   - {issue_type}: {len(issue_nums)} issues ({percentage:.1f}%)")
        print(f"     Examples: {issue_nums[:3]}")

print(f"\n4. TECHNICAL PROBLEMS:")
for problem_type, issues in sorted(results['technical_problems'].items(), 
                                  key=lambda x: len(x[1]), reverse=True):
    if issues:
        percentage = (len(issues) / results['total_issues']) * 100
        print(f"   - {problem_type}: {len(issues)} issues ({percentage:.1f}%)")
        # Print first 2 examples
        for issue in issues[:2]:
            print(f"     → Issue #{issue['number']}({issue['html_url']}): {issue['title'][:60]}...")

print(f"\n5. INFORMATION QUALITY:")
env_pct = (results['information_quality']['has_environment_info'] / results['total_issues']) * 100
err_pct = (results['information_quality']['has_error_messages'] / results['total_issues']) * 100
rep_pct = (results['information_quality']['has_reproduction_steps'] / results['total_issues']) * 100

print(f"   - Has environment info: {results['information_quality']['has_environment_info']} ({env_pct:.1f}%)")
print(f"   - Has error messages: {results['information_quality']['has_error_messages']} ({err_pct:.1f}%)")
print(f"   - Has reproduction steps: {results['information_quality']['has_reproduction_steps']} ({rep_pct:.1f}%)")
print(f"   - Well documented: {len(results['information_quality']['well_documented'])} issues")
print(f"   - Missing information: {len(results['information_quality']['missing_info'])} issues")

print(f"\n6. RESOLUTION PATTERNS:")
for pattern, issue_nums in results['resolution_patterns'].items():
    if issue_nums:
        percentage = (len(issue_nums) / results['state_distribution']['closed']) * 100 if results['state_distribution']['closed'] > 0 else 0
        print(f"   - {pattern}: {len(issue_nums)} issues ({percentage:.1f}% of closed)")
        if pattern == 'closed_as_stale':
            print(f"     Examples: {issue_nums[:3]}")

# Load issues again to print specific examples
with open('/root/yunwei37/vllm-exp/bug-study/analysis_llm/sample_results/vllm/label_based/bug/issues.json', 'r') as f:
    issues = json.load(f)

print("\n7. DETAILED EXAMPLES BY CATEGORY:")

# Model loading issues
print("\n   MODEL LOADING ISSUES:")
model_issues = [i for i in issues if i['number'] in results['issue_types'].get('model_loading', [])][:2]
for issue in model_issues:
    print(f"   → Issue #{issue['number']}({issue['html_url']})")
    print(f"     Title: {issue['title']}")
    print(f"     State: {issue['state']}, Comments: {issue['comments']}")

# GPU/Memory issues
print("\n   GPU/MEMORY ISSUES:")
gpu_issues = [i for i in issues if i['number'] in results['issue_types'].get('gpu_memory', [])][:2]
for issue in gpu_issues:
    print(f"   → Issue #{issue['number']}({issue['html_url']})")
    print(f"     Title: {issue['title']}")
    print(f"     State: {issue['state']}, Comments: {issue['comments']}")

# API/Serving issues
print("\n   API/SERVING ISSUES:")
api_issues = [i for i in issues if i['number'] in results['issue_types'].get('api_serving', [])][:2]
for issue in api_issues:
    print(f"   → Issue #{issue['number']}({issue['html_url']})")
    print(f"     Title: {issue['title']}")
    print(f"     State: {issue['state']}, Comments: {issue['comments']}")