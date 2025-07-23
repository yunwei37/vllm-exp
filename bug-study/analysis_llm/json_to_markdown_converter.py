#!/usr/bin/env python3
"""
Convert sample JSON files to simplified Markdown format for easier analysis
"""

import json
from pathlib import Path
from datetime import datetime


def simplify_issue(issue):
    """Simplify issue data by removing unnecessary fields"""
    return {
        'title': issue.get('title', ''),
        'body': issue.get('body', ''),
        'labels': issue.get('labels', []),
        'state': issue.get('state', ''),
        'created_at': issue.get('created_at', ''),
        'closed_at': issue.get('closed_at', ''),
        'comments': issue.get('comments', 0),
        'html_url': issue.get('html_url', '')
    }


def issue_to_markdown(issue):
    """Convert a single issue to markdown format"""
    md = f"## Issue #{issue.get('number', 'N/A')}: {issue.get('title', 'No Title')}\n\n"
    md += f"**Link**: {issue.get('html_url', 'N/A')}\n"
    md += f"**State**: {issue.get('state', 'unknown')}\n"
    md += f"**Created**: {issue.get('created_at', 'N/A')}\n"
    
    if issue.get('closed_at'):
        md += f"**Closed**: {issue.get('closed_at')}\n"
    
    md += f"**Comments**: {issue.get('comments', 0)}\n"
    
    # Labels
    labels = issue.get('labels', [])
    if labels:
        md += f"**Labels**: {', '.join(labels)}\n"
    
    md += "\n### Description\n\n"
    body = issue.get('body', 'No description provided.')
    if body:
        # Truncate very long bodies
        if len(body) > 1000:
            body = body[:1000] + "\n\n[... truncated for brevity ...]"
        md += body
    else:
        md += "No description provided."
    
    md += "\n\n---\n\n"
    return md


def convert_json_to_markdown(json_path, output_dir):
    """Convert a JSON file of issues to markdown"""
    with open(json_path, 'r') as f:
        issues = json.load(f)
    
    # Create output directory structure
    relative_path = json_path.relative_to(Path('/root/yunwei37/vllm-exp/bug-study/analysis_llm/sample_results'))
    output_path = output_dir / relative_path.parent / f"{relative_path.stem}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate markdown content
    md_content = f"# {relative_path.parent.name} - {relative_path.stem}\n\n"
    md_content += f"**Total Issues**: {len(issues)}\n"
    md_content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add summary statistics
    open_count = sum(1 for issue in issues if issue.get('state') == 'open')
    closed_count = len(issues) - open_count
    
    md_content += f"## Summary Statistics\n\n"
    md_content += f"- Open Issues: {open_count}\n"
    md_content += f"- Closed Issues: {closed_count}\n"
    
    # Label frequency
    all_labels = []
    for issue in issues:
        all_labels.extend(issue.get('labels', []))
    
    if all_labels:
        from collections import Counter
        label_counts = Counter(all_labels)
        md_content += f"\n### Label Distribution\n\n"
        for label, count in label_counts.most_common(10):
            md_content += f"- {label}: {count} issues\n"
    
    md_content += "\n---\n\n"
    
    # Convert each issue
    for issue in issues:
        simplified = simplify_issue(issue)
        md_content += issue_to_markdown(simplified)
    
    # Write markdown file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return output_path


def process_all_samples():
    """Process all sample JSON files"""
    sample_dir = Path('/root/yunwei37/vllm-exp/bug-study/analysis_llm/sample_results')
    output_dir = Path('/root/yunwei37/vllm-exp/bug-study/analysis_llm/markdown_samples')
    
    # Find all issues.json files
    json_files = list(sample_dir.rglob('issues.json'))
    
    print(f"Found {len(json_files)} JSON files to convert")
    
    converted_count = 0
    for json_file in json_files:
        try:
            output_path = convert_json_to_markdown(json_file, output_dir)
            print(f"✓ Converted: {json_file.relative_to(sample_dir)} -> {output_path.name}")
            converted_count += 1
        except Exception as e:
            print(f"✗ Error converting {json_file}: {e}")
    
    print(f"\n✅ Successfully converted {converted_count}/{len(json_files)} files")
    
    # Create index file
    create_index_file(output_dir)


def create_index_file(output_dir):
    """Create an index file listing all converted markdown files"""
    index_path = output_dir / 'INDEX.md'
    
    md_files = list(output_dir.rglob('*.md'))
    md_files = [f for f in md_files if f.name != 'INDEX.md']
    
    # Group by framework
    frameworks = {}
    for md_file in md_files:
        parts = md_file.relative_to(output_dir).parts
        if len(parts) > 0:
            framework = parts[0]
            if framework not in frameworks:
                frameworks[framework] = {}
            
            if len(parts) > 1:
                method = parts[1]
                if method not in frameworks[framework]:
                    frameworks[framework][method] = []
                frameworks[framework][method].append(md_file)
    
    # Generate index content
    index_content = "# Markdown Sample Index\n\n"
    index_content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    index_content += f"**Total Files**: {len(md_files)}\n\n"
    
    for framework in sorted(frameworks.keys()):
        index_content += f"## {framework}\n\n"
        
        for method in sorted(frameworks[framework].keys()):
            index_content += f"### {method}\n\n"
            
            for md_file in sorted(frameworks[framework][method]):
                relative_path = md_file.relative_to(output_dir)
                file_name = md_file.stem
                index_content += f"- [{file_name}]({relative_path})\n"
            
            index_content += "\n"
    
    with open(index_path, 'w') as f:
        f.write(index_content)
    
    print(f"\n📋 Index created at: {index_path}")


if __name__ == "__main__":
    process_all_samples()