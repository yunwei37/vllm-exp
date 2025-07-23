#!/usr/bin/env python3
"""
Create Analysis Result Directory Structure
Mirrors the sample_results structure and adds research question documentation
"""

import json
from pathlib import Path
from datetime import datetime


def create_analysis_structure():
    """Create analysis result directory structure with RQ documentation"""
    
    # Base directories
    sample_base = Path('/root/yunwei37/vllm-exp/bug-study/analysis_llm/sample_results')
    analysis_base = Path('/root/yunwei37/vllm-exp/bug-study/analysis_llm/analysis_results')
    
    # Research questions for each sampling method
    rq_mapping = {
        'label_based': {
            'title': 'Label-Based Analysis',
            'questions': [
                'RQ1: What types of issues are actually captured under this label?',
                'RQ2: What are the common technical problems in this label category?',
                'RQ3: How are issues in this category typically resolved?',
                'RQ4: What information is typically missing or well-provided?'
            ]
        },
        'temporal': {
            'title': 'Temporal Analysis',
            'questions': [
                'RQ1: How have issue types evolved over time?',
                'RQ2: What new problems emerged recently?',
                'RQ3: Which problems have persisted throughout the project?',
                'RQ4: Are there seasonal or release-related patterns?'
            ]
        },
        'resolution_time': {
            'title': 'Resolution Time Analysis',
            'questions': [
                'RQ1: What distinguishes quick vs slow resolution?',
                'RQ2: Are quickly closed issues actually fixed or dismissed?',
                'RQ3: Why do some issues remain open indefinitely?',
                'RQ4: What factors correlate with resolution speed?'
            ]
        },
        'complexity': {
            'title': 'Complexity Analysis',
            'questions': [
                'RQ1: What causes extensive discussion?',
                'RQ2: Are zero-comment issues low quality or self-evident?',
                'RQ3: Does discussion length correlate with issue difficulty?',
                'RQ4: What patterns exist in high-complexity issues?'
            ]
        },
        'author': {
            'title': 'Author-Based Analysis',
            'questions': [
                'RQ1: How does issue quality vary by author experience?',
                'RQ2: What problems do new users face vs experienced users?',
                'RQ3: Do team members report different types of issues?',
                'RQ4: What support do different user types need?'
            ]
        },
        'reaction': {
            'title': 'Reaction-Based Analysis',
            'questions': [
                'RQ1: What makes issues resonate with the community?',
                'RQ2: Are highly-reacted issues actually more important?',
                'RQ3: Why do some issues get no engagement?',
                'RQ4: Do reactions correlate with issue priority or resolution?'
            ]
        },
        'long_tail': {
            'title': 'Long-Tail Label Analysis',
            'questions': [
                'RQ1: Why are tail labels rarely used?',
                'RQ2: Do tail labels represent important edge cases?',
                'RQ3: Should tail labels be consolidated or removed?',
                'RQ4: What patterns exist in label frequency distribution?'
            ]
        },
        'cross_reference': {
            'title': 'Cross-Reference Analysis',
            'questions': [
                'RQ1: What patterns exist in related issue clusters?',
                'RQ2: How accurate are duplicate markings?',
                'RQ3: What issues spawn the most related work?',
                'RQ4: How do cross-references help understanding?'
            ]
        },
        'state_transition': {
            'title': 'State Transition Analysis',
            'questions': [
                'RQ1: Why do issues get reopened?',
                'RQ2: What causes label/state churn?',
                'RQ3: Are churning issues poorly defined?',
                'RQ4: What patterns exist in issue lifecycle?'
            ]
        },
        'anomaly': {
            'title': 'Anomaly Analysis',
            'questions': [
                'RQ1: What causes extreme metadata patterns?',
                'RQ2: Are anomalous issues special cases or noise?',
                'RQ3: Do format anomalies indicate quality issues?',
                'RQ4: What can we learn from outliers?'
            ]
        }
    }
    
    # Additional cross-cutting research questions
    cross_cutting_rqs = {
        'minor_labels': {
            'title': 'Minor Label Analysis',
            'questions': [
                'RQ5: Do minor labels represent distinct issue categories?',
                'RQ6: Why do these labels have low usage?',
                'RQ7: Should any be promoted to major labels or deprecated?',
                'RQ8: What patterns exist across minor labels?'
            ]
        },
        'cross_label': {
            'title': 'Cross-Label Analysis',
            'questions': [
                'RQ7: Which labels have overlapping content?',
                'RQ8: What categories exist in unlabeled issues?',
                'RQ9: How consistent is labeling across similar issues?',
                'RQ10: What label combinations are most effective?'
            ]
        }
    }
    
    def create_rq_document(output_dir, title, questions, sample_info=None):
        """Create research question document for a directory"""
        rq_content = f"""# {title}

## Research Questions

"""
        for q in questions:
            rq_content += f"- {q}\n"
        
        if sample_info:
            rq_content += f"""
## Sample Information

- **Framework**: {sample_info.get('framework', 'N/A')}
- **Method Type**: {sample_info.get('method_type', 'N/A')}
- **Sample Name**: {sample_info.get('sample_name', 'N/A')}
- **Sample Size**: {sample_info.get('sample_size', 'N/A')}
- **Sampling Criteria**: {sample_info.get('sampling_criteria', 'N/A')}
"""
        
        # Add LLM prompt instructions
        rq_content += f"""
## LLM Analysis Prompt Instructions

When analyzing these issues, please follow these guidelines:

### 1. Citation Requirements
- **ALWAYS cite the original issue** when discussing specific examples
- Use format: `Issue #[number]([html_url])` 
- Example: `Issue #1234(https://github.com/vllm-project/vllm/issues/1234)`
- When summarizing patterns, cite at least 3 representative issues

### 2. Analysis Structure
For each research question:
1. **Overall Pattern**: Describe the general pattern observed
2. **Specific Examples**: Cite 3-5 specific issues that exemplify this pattern
3. **Variations**: Note any significant variations or outliers (with citations)
4. **Quantification**: Provide rough percentages or counts where relevant

### 3. Issue Reference Format
When referencing issues in your analysis:
```
As seen in Issue #1234(link), the user reports [specific detail]...
Similar patterns appear in Issue #5678(link) and Issue #9012(link).
```

### 4. Summary Requirements
- Group similar issues together but maintain traceability
- For each finding, provide issue numbers as evidence
- Create categories/taxonomies with example issues for each category

### 5. Quality Checks
Before finalizing analysis:
- Ensure every claim is backed by specific issue citations
- Verify issue numbers and links are correct
- Check that patterns are supported by multiple examples

### Example Analysis Format:
```
RQ1: What types of issues are actually captured under this label?

**Finding 1: Memory-related errors (40% of samples)**
- Issue #1234(link): OOM during model loading
- Issue #5678(link): GPU memory fragmentation
- Issue #9012(link): Memory leak in attention mechanism

**Finding 2: Configuration problems (30% of samples)**
- Issue #2345(link): Incorrect tensor_parallel_size
- Issue #6789(link): Incompatible model format
[etc.]
```
"""
        
        rq_content += f"""
## Analysis Status

⏳ **Status**: Not Started
📅 **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Placeholder for Analysis Results

This directory will contain:
1. `analysis_results.json` - Structured analysis output
2. `findings.md` - Human-readable findings and insights
3. `patterns.json` - Extracted patterns and categories
4. `recommendations.md` - Actionable recommendations

---
*This is a placeholder document. Analysis has not been conducted yet.*
"""
        
        with open(output_dir / 'RESEARCH_QUESTIONS.md', 'w') as f:
            f.write(rq_content)
            
    def create_placeholder_files(output_dir):
        """Create placeholder files for analysis results"""
        # Create empty JSON files
        placeholders = {
            'analysis_results.json': {
                'status': 'not_started',
                'created_at': datetime.now().isoformat(),
                'results': {}
            },
            'patterns.json': {
                'status': 'not_started',
                'patterns': []
            }
        }
        
        for filename, content in placeholders.items():
            with open(output_dir / filename, 'w') as f:
                json.dump(content, f, indent=2)
        
        # Create markdown placeholders
        md_files = {
            'findings.md': '# Analysis Findings\n\n⏳ Analysis not yet conducted.\n',
            'recommendations.md': '# Recommendations\n\n⏳ Analysis not yet conducted.\n'
        }
        
        for filename, content in md_files.items():
            with open(output_dir / filename, 'w') as f:
                f.write(content)
    
    # Process each framework
    frameworks = ['vllm', 'sglang', 'llama_cpp']
    
    for framework in frameworks:
        print(f"\nCreating analysis structure for {framework}...")
        
        sample_framework_dir = sample_base / framework
        if not sample_framework_dir.exists():
            print(f"  ⚠️  Sample directory not found: {sample_framework_dir}")
            continue
            
        analysis_framework_dir = analysis_base / framework
        
        # Process each sampling method
        for method_dir in sample_framework_dir.iterdir():
            if not method_dir.is_dir() or method_dir.name == '__pycache__':
                continue
                
            method_name = method_dir.name
            print(f"  Processing {method_name}...")
            
            # Create corresponding analysis directory
            analysis_method_dir = analysis_framework_dir / method_name
            
            # Process each sample group
            for sample_dir in method_dir.iterdir():
                if not sample_dir.is_dir():
                    continue
                    
                sample_name = sample_dir.name
                analysis_sample_dir = analysis_method_dir / sample_name
                analysis_sample_dir.mkdir(parents=True, exist_ok=True)
                
                # Load sample metadata if available
                metadata_path = sample_dir / 'metadata.json'
                sample_info = None
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        sample_info = json.load(f)
                
                # Get appropriate RQs
                if method_name in rq_mapping:
                    rq_info = rq_mapping[method_name]
                    create_rq_document(
                        analysis_sample_dir, 
                        f"{rq_info['title']} - {sample_name}",
                        rq_info['questions'],
                        sample_info
                    )
                else:
                    # Default RQs for unknown methods
                    create_rq_document(
                        analysis_sample_dir,
                        f"{method_name.title()} Analysis - {sample_name}",
                        [
                            'RQ1: What patterns exist in this sample?',
                            'RQ2: What insights can be derived?',
                            'RQ3: How does this compare to other samples?',
                            'RQ4: What are the implications?'
                        ],
                        sample_info
                    )
                
                # Create placeholder files
                create_placeholder_files(analysis_sample_dir)
                
        # Create cross-cutting analysis directories
        print(f"  Creating cross-cutting analysis directories...")
        
        # Minor labels analysis
        minor_labels_dir = analysis_framework_dir / 'cross_cutting' / 'minor_labels'
        minor_labels_dir.mkdir(parents=True, exist_ok=True)
        create_rq_document(
            minor_labels_dir,
            f"Minor Labels Analysis - {framework}",
            cross_cutting_rqs['minor_labels']['questions']
        )
        create_placeholder_files(minor_labels_dir)
        
        # Cross-label analysis
        cross_label_dir = analysis_framework_dir / 'cross_cutting' / 'cross_label'
        cross_label_dir.mkdir(parents=True, exist_ok=True)
        create_rq_document(
            cross_label_dir,
            f"Cross-Label Analysis - {framework}",
            cross_cutting_rqs['cross_label']['questions']
        )
        create_placeholder_files(cross_label_dir)
        
        # Create framework summary
        framework_summary = f"""# {framework} Analysis Summary

## Analysis Structure

This directory contains LLM-based analysis results organized by sampling method:

- **label_based/**: Analysis of each label category
- **temporal/**: Time-based patterns and evolution
- **resolution_time/**: Resolution speed patterns
- **complexity/**: Issue complexity analysis
- **author/**: Author-based patterns
- **reaction/**: Community engagement analysis
- **long_tail/**: Label frequency distribution analysis
- **cross_reference/**: Issue relationship analysis
- **state_transition/**: Issue lifecycle patterns
- **anomaly/**: Outlier and edge case analysis
- **cross_cutting/**: Cross-method analyses

## Progress Tracking

Total sample groups: {sum(1 for _ in (analysis_framework_dir).rglob('RESEARCH_QUESTIONS.md'))}
Completed: 0
In Progress: 0
Not Started: All

## Next Steps

1. Run LLM analysis on each sample group
2. Extract patterns and insights
3. Generate findings and recommendations
4. Create cross-cutting synthesis

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(analysis_framework_dir / 'README.md', 'w') as f:
            f.write(framework_summary)
    
    # Create overall summary
    overall_summary = f"""# LLM Analysis Results

## Overview

This directory contains the results of LLM-based analysis on sampled issues from three LLM serving frameworks:

1. **vLLM**: {sum(1 for _ in (analysis_base / 'vllm').rglob('RESEARCH_QUESTIONS.md')) if (analysis_base / 'vllm').exists() else 0} analysis groups
2. **SGLang**: {sum(1 for _ in (analysis_base / 'sglang').rglob('RESEARCH_QUESTIONS.md')) if (analysis_base / 'sglang').exists() else 0} analysis groups
3. **llama.cpp**: {sum(1 for _ in (analysis_base / 'llama_cpp').rglob('RESEARCH_QUESTIONS.md')) if (analysis_base / 'llama_cpp').exists() else 0} analysis groups

## Analysis Methods

Each framework has been analyzed using 10 different sampling methods:
- Label-based sampling (one group per label)
- Temporal sampling (time periods)
- Resolution time sampling (speed of closure)
- Complexity sampling (discussion levels)
- Author-based sampling (contributor types)
- Reaction-based sampling (community engagement)
- Long-tail sampling (label frequency)
- Cross-reference sampling (issue relationships)
- State transition sampling (lifecycle patterns)
- Anomaly sampling (outliers)

## Directory Structure

```
analysis_results/
├── vllm/
│   ├── [sampling_method]/
│   │   └── [sample_group]/
│   │       ├── RESEARCH_QUESTIONS.md
│   │       ├── analysis_results.json
│   │       ├── patterns.json
│   │       ├── findings.md
│   │       └── recommendations.md
│   └── cross_cutting/
├── sglang/
│   └── [same structure]
└── llama_cpp/
    └── [same structure]
```

## Research Questions

Each analysis group addresses specific research questions relevant to its sampling method.
See individual `RESEARCH_QUESTIONS.md` files for details.

## Status

⏳ All analyses are currently pending.

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(analysis_base / 'README.md', 'w') as f:
        f.write(overall_summary)
    
    print(f"\n✅ Analysis structure created successfully!")
    print(f"📁 Location: {analysis_base}")
    
    # Count created directories
    total_dirs = sum(1 for _ in analysis_base.rglob('RESEARCH_QUESTIONS.md'))
    print(f"📊 Total analysis groups created: {total_dirs}")


if __name__ == "__main__":
    create_analysis_structure()