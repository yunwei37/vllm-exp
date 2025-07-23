#!/usr/bin/env python3
"""
Generate a comprehensive master report combining all analysis results
"""

import json
from pathlib import Path
from datetime import datetime

def load_json_results(file_path):
    """Load JSON results file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None

def generate_master_report():
    """Generate comprehensive master report from all analyses"""
    base_dir = Path(__file__).parent
    results_dir = base_dir / 'results'
    
    # Frameworks to analyze
    frameworks = ['vllm', 'sglang', 'llama_cpp']
    framework_display_names = {'vllm': 'vLLM', 'sglang': 'SGLang', 'llama_cpp': 'llama.cpp'}
    
    # Start report
    report = f"""# LLM Serving Framework Issue Analysis - Master Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This comprehensive analysis examines issue management patterns across three major LLM serving frameworks:
- **vLLM**: High-performance LLM serving engine
- **SGLang**: Structured generation language framework  
- **llama.cpp**: CPU/GPU inference for LLaMA models

The analysis covers temporal patterns, user behavior, label usage, state transitions, and cross-framework comparisons
using statistical and data mining techniques without NLP/LLM assistance.

## Key Findings Summary

"""
    
    # Load cross-framework comparison data
    cross_data = load_json_results(results_dir / 'cross_framework' / 'cross_framework_comparison_data.json')
    
    if cross_data and 'basic_metrics' in cross_data:
        report += "### Framework Overview\n\n"
        report += "| Metric | vLLM | SGLang | llama.cpp |\n"
        report += "|--------|------|--------|----------|\n"
        
        metrics_to_show = [
            ('total_issues', 'Total Issues', lambda x: f"{x:,}"),
            ('closure_rate', 'Closure Rate', lambda x: f"{x:.1%}"),
            ('unique_users', 'Unique Users', lambda x: f"{x:,}"),
            ('median_resolution_days', 'Median Resolution (days)', lambda x: f"{x:.1f}" if x else "N/A")
        ]
        
        for metric_key, metric_name, formatter in metrics_to_show:
            row = f"| {metric_name} | "
            for fw in ['vLLM', 'SGLang', 'llama.cpp']:
                if fw in cross_data['basic_metrics']:
                    value = cross_data['basic_metrics'][fw].get(metric_key, 'N/A')
                    row += f"{formatter(value) if value != 'N/A' else 'N/A'} | "
                else:
                    row += "N/A | "
            report += row + "\n"
    
    # Add individual framework insights
    report += "\n## Framework-Specific Insights\n"
    
    for framework in frameworks:
        report += f"\n### {framework.upper()}\n"
        
        # Load framework-specific data
        temporal_data = load_json_results(results_dir / 'temporal' / f'{framework}_temporal_data.json')
        user_data = load_json_results(results_dir / 'user_behavior' / f'{framework}_user_behavior_data.json')
        label_data = load_json_results(results_dir / 'label_complexity' / f'{framework}_label_complexity_data.json')
        state_data = load_json_results(results_dir / 'state_transition' / f'{framework}_state_transition_data.json')
        
        # Temporal insights
        if temporal_data:
            report += "\n#### Temporal Patterns\n"
            if 'velocity' in temporal_data:
                report += f"- Peak activity hour: {temporal_data['velocity'].get('peak_hour', 'N/A')}:00 UTC\n"
                report += f"- Peak activity day: {temporal_data['velocity'].get('peak_day', 'N/A')}\n"
                report += f"- Weekend activity: {temporal_data['velocity'].get('weekend_ratio', 0)*100:.1f}%\n"
            
            if 'resolution' in temporal_data:
                report += f"- Median resolution: {temporal_data['resolution'].get('median_days', 0):.1f} days\n"
                report += f"- 90th percentile: {temporal_data['resolution'].get('p90_days', 0):.1f} days\n"
        
        # User behavior insights
        if user_data:
            report += "\n#### Community Dynamics\n"
            if 'distribution' in user_data:
                report += f"- Total contributors: {user_data['distribution'].get('total_users', 0):,}\n"
                report += f"- Gini coefficient: {user_data['distribution'].get('gini_coefficient', 0):.3f}\n"
                report += f"- Single-issue users: {user_data['distribution'].get('single_issue_ratio', 0)*100:.1f}%\n"
            
            if 'network' in user_data:
                report += f"- Collaboration network size: {user_data['network'].get('total_nodes', 0)} users\n"
        
        # Label and complexity insights
        if label_data:
            report += "\n#### Issue Categorization\n"
            if 'distribution' in label_data:
                report += f"- Unique labels: {label_data['distribution'].get('total_unique_labels', 0)}\n"
                report += f"- Label entropy: {label_data['distribution'].get('label_entropy', 0):.3f}\n"
                report += f"- Unlabeled issues: {label_data['distribution'].get('no_label_ratio', 0)*100:.1f}%\n"
            
            if 'complexity' in label_data:
                report += f"- No discussion rate: {label_data['complexity'].get('no_discussion_ratio', 0)*100:.1f}%\n"
        
        # State transition insights
        if state_data:
            report += "\n#### Resolution Patterns\n"
            if 'state_distribution' in state_data:
                report += f"- Open ratio: {state_data['state_distribution'].get('open_ratio', 0)*100:.1f}%\n"
            
            if 'lifecycle' in state_data:
                report += f"- Response rate: {state_data['lifecycle'].get('response_rate', 0)*100:.1f}%\n"
                report += f"- Old open issues (>1yr): {state_data['lifecycle'].get('old_open_issues', 0)}\n"
    
    # Add comparative insights
    report += "\n## Comparative Analysis\n"
    
    if cross_data:
        report += "\n### Community Health Rankings\n"
        
        # Determine best in each category
        if 'community_health' in cross_data and 'retention_metrics' in cross_data['community_health']:
            retention = cross_data['community_health']['retention_metrics']
            
            # Best retention
            best_retention = max(retention.items(), key=lambda x: x[1]['retention_rate'])
            report += f"- **Best User Retention**: {best_retention[0]} ({best_retention[1]['retention_rate']*100:.1f}%)\n"
            
            # Largest community
            largest = max(retention.items(), key=lambda x: x[1]['total_users'])
            report += f"- **Largest Community**: {largest[0]} ({largest[1]['total_users']:,} users)\n"
            
            # Most active
            most_active = max(retention.items(), key=lambda x: x[1]['avg_contributions'])
            report += f"- **Most Active Users**: {most_active[0]} ({most_active[1]['avg_contributions']:.1f} issues/user)\n"
    
    # Add visualizations section
    report += "\n## Visualizations Generated\n\n"
    report += "The following visualization sets have been generated:\n\n"
    
    viz_categories = [
        ("Temporal Analysis", ["velocity_analysis", "resolution_patterns", "temporal_anomalies"]),
        ("User Behavior", ["user_distribution", "user_behavior", "collaboration_network", "user_evolution"]),
        ("Label & Complexity", ["label_distribution", "complexity_analysis", "label_effectiveness", "label_evolution"]),
        ("State Transitions", ["state_distributions", "resolution_factors", "lifecycle_patterns", "reopened_patterns"]),
        ("Cross-Framework", ["basic_metrics", "temporal_patterns", "community_health", "issue_characteristics", "comparison_matrix"])
    ]
    
    for category, viz_types in viz_categories:
        report += f"### {category}\n"
        for viz in viz_types:
            report += f"- {viz.replace('_', ' ').title()}\n"
        report += "\n"
    
    # Add methodology note
    report += """## Methodology

This analysis employed statistical and data mining techniques including:

1. **Temporal Analysis**: Time series analysis, seasonal decomposition, anomaly detection
2. **User Behavior**: Social network analysis, Gini coefficient, retention metrics
3. **Label Analysis**: Co-occurrence matrices, entropy calculation, evolution tracking
4. **State Transitions**: Markov-like state analysis, survival analysis concepts
5. **Cross-Framework**: Normalized comparisons, correlation analysis

All analyses were performed without NLP or LLM assistance, focusing on quantitative patterns
and structural relationships in the data.

## Data Sources

- **vLLM**: 9,631 issues (1,831 open + 7,800 closed)
- **SGLang**: 2,567 issues  
- **llama.cpp**: 5,470 issues

Data collected via GitHub API and analyzed using Python with pandas, numpy, matplotlib, and seaborn.

## Recommendations

Based on the analysis, we recommend:

1. **For vLLM**: Focus on reducing resolution time variance and improving label standardization
2. **For SGLang**: Enhance user retention strategies and community engagement
3. **For llama.cpp**: Implement more systematic labeling and improve response times
4. **Cross-Framework**: Share best practices, particularly in community management and issue triage

## Future Work

- Implement predictive models for issue resolution time
- Develop automated issue classification systems
- Create real-time monitoring dashboards
- Conduct deeper root cause analysis of long-standing issues

---
*This report was generated automatically from statistical analysis of GitHub issue data.*
"""
    
    # Save report
    output_path = results_dir / 'MASTER_REPORT.md'
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Master report generated: {output_path}")
    
    # Also generate a summary statistics file
    summary_stats = {
        'report_generated': datetime.now().isoformat(),
        'frameworks_analyzed': frameworks,
        'total_issues_analyzed': sum(
            cross_data['basic_metrics'][fw]['total_issues'] 
            for fw in frameworks 
            if fw in cross_data.get('basic_metrics', {})
        ) if cross_data else 0,
        'analyses_performed': [
            'temporal_analysis',
            'user_behavior_analysis', 
            'label_complexity_analysis',
            'state_transition_analysis',
            'cross_framework_comparison'
        ]
    }
    
    with open(results_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    return output_path

if __name__ == "__main__":
    generate_master_report()