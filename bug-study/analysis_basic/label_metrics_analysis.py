#!/usr/bin/env python3
"""
Label Metrics Analysis: Generate sortable table with average issue length, 
comment count, and resolve time grouped by label
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class LabelMetricsAnalyzer:
    def __init__(self, framework_name, data_path):
        self.framework = framework_name
        self.data_path = Path(data_path)
        self.df = None
        self.label_metrics = {}
        
    def load_data(self):
        """Load and preprocess issue data"""
        print(f"\nLoading {self.framework} data...")
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        # Extract items from different JSON structures
        if isinstance(data, dict):
            items = data.get('items', data.get('issues', []))
        else:
            items = data
            
        self.df = pd.DataFrame(items)
        print(f"Loaded {len(self.df)} issues")
        
        # Convert timestamps
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
        self.df['closed_at'] = pd.to_datetime(self.df['closed_at'], errors='coerce')
        
        # Extract label names
        self.df['label_names'] = self.df['labels'].apply(
            lambda x: [l['name'] for l in x if isinstance(l, dict) and 'name' in l] 
            if isinstance(x, list) else []
        )
        
        # Calculate issue body length (characters)
        self.df['body_length'] = self.df['body'].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        
        # Extract comment count (ensure it's numeric)
        self.df['comments'] = pd.to_numeric(self.df['comments'], errors='coerce').fillna(0)
        
        # Calculate resolution time in days
        self.df['is_closed'] = self.df['state'] == 'closed'
        self.df['resolution_days'] = pd.NaT
        mask = self.df['is_closed'] & self.df['closed_at'].notna()
        self.df.loc[mask, 'resolution_days'] = (
            self.df.loc[mask, 'closed_at'] - self.df.loc[mask, 'created_at']
        ).dt.total_seconds() / 86400
        
    def calculate_label_metrics(self):
        """Calculate metrics for each label"""
        print(f"\nCalculating metrics for {self.framework}...")
        
        # Get all unique labels
        all_labels = set()
        for labels in self.df['label_names']:
            all_labels.update(labels)
        
        # Calculate metrics for each label
        for label in sorted(all_labels):
            # Filter issues with this label
            mask = self.df['label_names'].apply(lambda x: label in x)
            label_issues = self.df[mask]
            
            if len(label_issues) == 0:
                continue
                
            # Calculate metrics
            metrics = {
                'label': label,
                'issue_count': len(label_issues),
                'avg_body_length': label_issues['body_length'].mean(),
                'median_body_length': label_issues['body_length'].median(),
                'avg_comment_count': label_issues['comments'].mean(),
                'median_comment_count': label_issues['comments'].median(),
                'closed_count': label_issues['is_closed'].sum(),
                'closure_rate': label_issues['is_closed'].mean() * 100,
            }
            
            # Resolution time stats (only for closed issues)
            closed_with_time = label_issues[label_issues['resolution_days'].notna()]
            if len(closed_with_time) > 0:
                metrics['avg_resolution_days'] = closed_with_time['resolution_days'].mean()
                metrics['median_resolution_days'] = closed_with_time['resolution_days'].median()
                metrics['min_resolution_days'] = closed_with_time['resolution_days'].min()
                metrics['max_resolution_days'] = closed_with_time['resolution_days'].max()
            else:
                metrics['avg_resolution_days'] = None
                metrics['median_resolution_days'] = None
                metrics['min_resolution_days'] = None
                metrics['max_resolution_days'] = None
            
            self.label_metrics[label] = metrics
    
    def generate_sortable_table(self, sort_by='issue_count', ascending=False):
        """Generate a sortable table of label metrics"""
        if not self.label_metrics:
            print("No label metrics calculated yet!")
            return None
            
        # Convert to DataFrame for easy sorting
        df_metrics = pd.DataFrame.from_dict(self.label_metrics, orient='index')
        
        # Sort by specified column
        if sort_by in df_metrics.columns:
            df_metrics = df_metrics.sort_values(by=sort_by, ascending=ascending)
        
        # Prepare display columns
        display_columns = [
            'label',
            'issue_count',
            'avg_body_length',
            'avg_comment_count',
            'avg_resolution_days',
            'median_resolution_days',
            'closure_rate'
        ]
        
        # Format the data for display
        display_data = []
        for _, row in df_metrics.iterrows():
            display_row = {
                'Label': row['label'],
                'Issues': int(row['issue_count']),
                'Avg Body Length': f"{row['avg_body_length']:.0f}" if pd.notna(row['avg_body_length']) else 'N/A',
                'Avg Comments': f"{row['avg_comment_count']:.1f}" if pd.notna(row['avg_comment_count']) else 'N/A',
                'Avg Resolution (days)': f"{row['avg_resolution_days']:.1f}" if pd.notna(row['avg_resolution_days']) else 'N/A',
                'Median Resolution (days)': f"{row['median_resolution_days']:.1f}" if pd.notna(row['median_resolution_days']) else 'N/A',
                'Closure Rate': f"{row['closure_rate']:.1f}%" if pd.notna(row['closure_rate']) else 'N/A'
            }
            display_data.append(display_row)
        
        return pd.DataFrame(display_data), df_metrics
    
    def print_analysis(self):
        """Print comprehensive analysis"""
        print(f"\n{'='*80}")
        print(f"{self.framework} - Label Metrics Analysis")
        print(f"{'='*80}")
        
        # Overall statistics
        print(f"\nOverall Statistics:")
        print(f"  Total Issues: {len(self.df)}")
        print(f"  Unique Labels: {len(self.label_metrics)}")
        print(f"  Issues with Labels: {self.df['label_names'].apply(len).gt(0).sum()}")
        print(f"  Avg Labels per Issue: {self.df['label_names'].apply(len).mean():.2f}")
        
        # Print different sorted views
        sort_options = [
            ('issue_count', False, 'Most Used Labels'),
            ('avg_body_length', False, 'Labels by Average Issue Length'),
            ('avg_comment_count', False, 'Labels by Average Comment Count'),
            ('avg_resolution_days', True, 'Labels by Fastest Resolution Time'),
            ('closure_rate', False, 'Labels by Highest Closure Rate')
        ]
        
        for sort_col, asc, title in sort_options:
            print(f"\n{title}:")
            print("-" * 80)
            display_df, _ = self.generate_sortable_table(sort_by=sort_col, ascending=asc)
            if display_df is not None and len(display_df) > 0:
                # Show top 10
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', 20)
                print(display_df.head(10).to_string(index=False))
    
    def save_results(self, output_path):
        """Save results to CSV for further analysis"""
        if not self.label_metrics:
            print("No metrics to save!")
            return
            
        df_metrics = pd.DataFrame.from_dict(self.label_metrics, orient='index')
        df_metrics.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    def create_visualizations(self, output_dir):
        """Create comprehensive visualizations for label metrics"""
        if not self.label_metrics:
            print("No metrics to visualize!")
            return
            
        # Ensure output directory exists
        vis_dir = Path(output_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert metrics to DataFrame
        df_metrics = pd.DataFrame.from_dict(self.label_metrics, orient='index')
        df_metrics = df_metrics[df_metrics['issue_count'] >= 10]  # Filter labels with at least 10 issues
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. All Labels by Issue Count
        fig, ax = plt.subplots(figsize=(12, max(8, len(df_metrics) * 0.3)))
        sorted_labels = df_metrics.sort_values('issue_count', ascending=False)
        bars = ax.barh(range(len(sorted_labels)), sorted_labels['issue_count'], color='skyblue', edgecolor='navy')
        ax.set_yticks(range(len(sorted_labels)))
        ax.set_yticklabels(sorted_labels['label'].values)
        ax.set_xlabel('Number of Issues', fontsize=12)
        ax.set_title(f'{self.framework} - All Labels by Issue Count', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(sorted_labels.iterrows()):
            ax.text(row['issue_count'] + 20, i, f"{int(row['issue_count'])}", 
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(vis_dir / f'{self.framework.lower()}_top_labels.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Average Body Length by Label
        fig, ax = plt.subplots(figsize=(12, max(8, len(df_metrics) * 0.3)))
        sorted_by_length = df_metrics.sort_values('avg_body_length', ascending=False)
        bars = ax.barh(range(len(sorted_by_length)), sorted_by_length['avg_body_length'], color='coral', edgecolor='darkred')
        ax.set_yticks(range(len(sorted_by_length)))
        ax.set_yticklabels(sorted_by_length['label'].values)
        ax.set_xlabel('Average Body Length (characters)', fontsize=12)
        ax.set_title(f'{self.framework} - All Labels by Average Issue Length', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (idx, row) in enumerate(sorted_by_length.iterrows()):
            ax.text(row['avg_body_length'] + 200, i, f"{int(row['avg_body_length'])}", 
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(vis_dir / f'{self.framework.lower()}_avg_body_length.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Average Comment Count by Label
        fig, ax = plt.subplots(figsize=(12, max(8, len(df_metrics) * 0.3)))
        sorted_by_comments = df_metrics.sort_values('avg_comment_count', ascending=False)
        bars = ax.barh(range(len(sorted_by_comments)), sorted_by_comments['avg_comment_count'], color='lightgreen', edgecolor='darkgreen')
        ax.set_yticks(range(len(sorted_by_comments)))
        ax.set_yticklabels(sorted_by_comments['label'].values)
        ax.set_xlabel('Average Comment Count', fontsize=12)
        ax.set_title(f'{self.framework} - All Labels by Average Comment Count', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (idx, row) in enumerate(sorted_by_comments.iterrows()):
            ax.text(row['avg_comment_count'] + 0.2, i, f"{row['avg_comment_count']:.1f}", 
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(vis_dir / f'{self.framework.lower()}_avg_comments.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Resolution Time Analysis (only for labels with resolution data)
        df_with_resolution = df_metrics[df_metrics['avg_resolution_days'].notna()]
        if len(df_with_resolution) > 0:
            fig, ax = plt.subplots(figsize=(12, max(8, len(df_with_resolution) * 0.3)))
            sorted_by_resolution = df_with_resolution.sort_values('avg_resolution_days', ascending=True)
            bars = ax.barh(range(len(sorted_by_resolution)), sorted_by_resolution['avg_resolution_days'], 
                          color='gold', edgecolor='darkorange')
            ax.set_yticks(range(len(sorted_by_resolution)))
            ax.set_yticklabels(sorted_by_resolution['label'].values)
            ax.set_xlabel('Average Resolution Time (days)', fontsize=12)
            ax.set_title(f'{self.framework} - All Labels by Resolution Time (Fastest First)', fontsize=14, fontweight='bold')
            
            # Add value labels
            for i, (idx, row) in enumerate(sorted_by_resolution.iterrows()):
                ax.text(row['avg_resolution_days'] + 1, i, f"{row['avg_resolution_days']:.1f}", 
                       va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(vis_dir / f'{self.framework.lower()}_resolution_time.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Closure Rate by Label
        fig, ax = plt.subplots(figsize=(12, max(8, len(df_metrics) * 0.3)))
        sorted_by_closure = df_metrics.sort_values('closure_rate', ascending=False)
        bars = ax.barh(range(len(sorted_by_closure)), sorted_by_closure['closure_rate'], color='mediumpurple', edgecolor='purple')
        ax.set_yticks(range(len(sorted_by_closure)))
        ax.set_yticklabels(sorted_by_closure['label'].values)
        ax.set_xlabel('Closure Rate (%)', fontsize=12)
        ax.set_title(f'{self.framework} - All Labels by Closure Rate', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 105)
        
        # Add value labels
        for i, (idx, row) in enumerate(sorted_by_closure.iterrows()):
            ax.text(row['closure_rate'] + 1, i, f"{row['closure_rate']:.1f}%", 
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(vis_dir / f'{self.framework.lower()}_closure_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Combined Metrics Scatter Plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.framework} - Label Metrics Correlations', fontsize=16, fontweight='bold')
        
        # Scatter: Issue Count vs Avg Body Length
        scatter1 = ax1.scatter(df_metrics['issue_count'], df_metrics['avg_body_length'], 
                              s=100, alpha=0.6, c=df_metrics['closure_rate'], cmap='RdYlGn')
        ax1.set_xlabel('Issue Count', fontsize=12)
        ax1.set_ylabel('Avg Body Length', fontsize=12)
        ax1.set_title('Issue Count vs Average Body Length', fontsize=14)
        
        # Add labels for top points
        top_issues = df_metrics.nlargest(5, 'issue_count')
        for idx, row in top_issues.iterrows():
            ax1.annotate(row['label'], (row['issue_count'], row['avg_body_length']), 
                        fontsize=9, alpha=0.7)
        
        # Scatter: Issue Count vs Avg Comments
        ax2.scatter(df_metrics['issue_count'], df_metrics['avg_comment_count'], 
                   s=100, alpha=0.6, c=df_metrics['closure_rate'], cmap='RdYlGn')
        ax2.set_xlabel('Issue Count', fontsize=12)
        ax2.set_ylabel('Avg Comment Count', fontsize=12)
        ax2.set_title('Issue Count vs Average Comments', fontsize=14)
        
        # Scatter: Avg Body Length vs Resolution Time
        df_res = df_metrics[df_metrics['avg_resolution_days'].notna()]
        if len(df_res) > 0:
            ax3.scatter(df_res['avg_body_length'], df_res['avg_resolution_days'], 
                       s=100, alpha=0.6, c=df_res['closure_rate'], cmap='RdYlGn')
            ax3.set_xlabel('Avg Body Length', fontsize=12)
            ax3.set_ylabel('Avg Resolution Days', fontsize=12)
            ax3.set_title('Body Length vs Resolution Time', fontsize=14)
        
        # Scatter: Comments vs Resolution Time
        if len(df_res) > 0:
            ax4.scatter(df_res['avg_comment_count'], df_res['avg_resolution_days'], 
                       s=100, alpha=0.6, c=df_res['closure_rate'], cmap='RdYlGn')
            ax4.set_xlabel('Avg Comment Count', fontsize=12)
            ax4.set_ylabel('Avg Resolution Days', fontsize=12)
            ax4.set_title('Comments vs Resolution Time', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(scatter1, ax=ax4, label='Closure Rate (%)')
        
        plt.tight_layout()
        plt.savefig(vis_dir / f'{self.framework.lower()}_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {vis_dir}")


def analyze_all_frameworks():
    """Analyze all three frameworks"""
    frameworks = [
        ('vLLM', '../data/vllm_all_issues.json'),
        ('SGLang', '../data/sglang_issues.json'),
        ('llama.cpp', '../data/llama_cpp_issues.json')
    ]
    
    # Create main output directory
    output_base = Path("results/label_metrics")
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for framework, data_path in frameworks:
        try:
            analyzer = LabelMetricsAnalyzer(framework, data_path)
            analyzer.load_data()
            analyzer.calculate_label_metrics()
            analyzer.print_analysis()
            
            # Create framework-specific subdirectory
            framework_dir = output_base / framework.lower().replace('.', '_')
            framework_dir.mkdir(exist_ok=True)
            
            # Save CSV results
            csv_path = framework_dir / f"{framework.lower().replace('.', '_')}_metrics.csv"
            analyzer.save_results(csv_path)
            
            # Create visualizations
            vis_dir = framework_dir / "visualizations"
            analyzer.create_visualizations(vis_dir)
            
            all_results[framework] = analyzer.label_metrics
            
        except Exception as e:
            print(f"\nError analyzing {framework}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


if __name__ == "__main__":
    # Run analysis
    results = analyze_all_frameworks()
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nResults have been saved to the 'results/label_metrics' directory.")
    print("Each framework has its own subdirectory containing:")
    print("  - CSV file with all label metrics")
    print("  - Visualizations folder with multiple charts")
    print("\nYou can open the CSV files in Excel or any spreadsheet software to:")
    print("  - Sort by any column")
    print("  - Filter by specific labels")
    print("  - Create custom visualizations")