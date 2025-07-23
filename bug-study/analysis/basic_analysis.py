#!/usr/bin/env python3
"""
Basic statistical and data mining analysis for LLM serving framework issues.
Analyzes vLLM, SGLang, and llama.cpp issue data without using NLP/LLM techniques.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class IssueAnalyzer:
    def __init__(self, framework_name, data_path):
        self.framework = framework_name
        self.data_path = Path(data_path)
        self.data = None
        self.df = None
        self.results = {}
        
    def load_data(self):
        """Load JSON data and convert to DataFrame"""
        print(f"\nLoading {self.framework} data...")
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        
        # Extract items
        items = self.data.get('items', self.data if isinstance(self.data, list) else [])
        self.df = pd.DataFrame(items)
        print(f"Loaded {len(self.df)} issues")
        
        # Convert timestamps
        for col in ['created_at', 'updated_at', 'closed_at']:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Extract additional fields
        self.df['is_closed'] = self.df['state'] == 'closed'
        self.df['has_comments'] = self.df['comments'] > 0
        self.df['label_count'] = self.df['labels'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Calculate resolution time
        mask = self.df['is_closed'] & self.df['closed_at'].notna()
        self.df.loc[mask, 'resolution_hours'] = (
            self.df.loc[mask, 'closed_at'] - self.df.loc[mask, 'created_at']
        ).dt.total_seconds() / 3600
        
        # Extract time components
        self.df['created_hour'] = self.df['created_at'].dt.hour
        self.df['created_dow'] = self.df['created_at'].dt.dayofweek
        self.df['created_month'] = self.df['created_at'].dt.to_period('M')
        
    def temporal_analysis(self):
        """Q1: Temporal Analysis"""
        print(f"\n[{self.framework}] Running temporal analysis...")
        results = {}
        
        # Q1.1: Issue volume trends
        monthly_counts = self.df.groupby('created_month').size()
        results['monthly_issue_counts'] = monthly_counts.to_dict()
        results['avg_issues_per_month'] = monthly_counts.mean()
        results['trend_direction'] = 'increasing' if monthly_counts[-6:].mean() > monthly_counts.mean() else 'decreasing'
        
        # Q1.2: Resolution time patterns
        resolution_times = self.df['resolution_hours'].dropna()
        if len(resolution_times) > 0:
            results['resolution_stats'] = {
                'mean_hours': resolution_times.mean(),
                'median_hours': resolution_times.median(),
                'p90_hours': resolution_times.quantile(0.9),
                'p99_hours': resolution_times.quantile(0.99),
                'min_hours': resolution_times.min(),
                'max_hours': resolution_times.max()
            }
        
        # Q1.3: Activity patterns
        hourly_dist = self.df['created_hour'].value_counts().sort_index()
        dow_dist = self.df['created_dow'].value_counts().sort_index()
        results['peak_hours'] = hourly_dist.nlargest(3).index.tolist()
        results['peak_days'] = dow_dist.nlargest(3).index.tolist()
        
        # Q1.4: Response time (first update)
        response_times = []
        for _, row in self.df.iterrows():
            if pd.notna(row['updated_at']) and row['updated_at'] != row['created_at']:
                response_hours = (row['updated_at'] - row['created_at']).total_seconds() / 3600
                if response_hours > 0:  # Avoid negative or zero times
                    response_times.append(response_hours)
        
        if response_times:
            results['response_time_stats'] = {
                'mean_hours': np.mean(response_times),
                'median_hours': np.median(response_times),
                'p90_hours': np.percentile(response_times, 90)
            }
        
        self.results['temporal'] = results
        
    def label_analysis(self):
        """Q2: Label and Category Analysis"""
        print(f"\n[{self.framework}] Running label analysis...")
        results = {}
        
        # Extract all labels
        all_labels = []
        for labels in self.df['labels']:
            if isinstance(labels, list):
                all_labels.extend([l['name'] for l in labels if isinstance(l, dict) and 'name' in l])
        
        # Q2.1: Label distribution
        label_counts = Counter(all_labels)
        results['top_labels'] = dict(label_counts.most_common(20))
        results['total_unique_labels'] = len(label_counts)
        
        # Q2.2: Label co-occurrence
        label_pairs = defaultdict(int)
        for labels in self.df['labels']:
            if isinstance(labels, list):
                label_names = [l['name'] for l in labels if isinstance(l, dict) and 'name' in l]
                for i in range(len(label_names)):
                    for j in range(i+1, len(label_names)):
                        pair = tuple(sorted([label_names[i], label_names[j]]))
                        label_pairs[pair] += 1
        
        results['top_label_pairs'] = dict(sorted(label_pairs.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Q2.3: Label-resolution correlation
        label_resolution = {}
        for label, count in label_counts.most_common(10):
            mask = self.df['labels'].apply(lambda x: any(l.get('name') == label for l in x if isinstance(l, dict)))
            labeled_issues = self.df[mask]
            if len(labeled_issues) > 0:
                resolution_rate = labeled_issues['is_closed'].mean()
                avg_resolution_time = labeled_issues['resolution_hours'].mean() if 'resolution_hours' in labeled_issues else None
                label_resolution[label] = {
                    'count': len(labeled_issues),
                    'resolution_rate': resolution_rate,
                    'avg_resolution_hours': avg_resolution_time
                }
        
        results['label_resolution_stats'] = label_resolution
        self.results['labels'] = results
        
    def user_analysis(self):
        """Q3: User and Contributor Analysis"""
        print(f"\n[{self.framework}] Running user analysis...")
        results = {}
        
        # Extract user info
        self.df['user_login'] = self.df['user'].apply(lambda x: x.get('login') if isinstance(x, dict) else None)
        
        # Q3.1: User activity distribution
        user_counts = self.df['user_login'].value_counts()
        results['total_unique_users'] = len(user_counts)
        results['top_contributors'] = user_counts.head(20).to_dict()
        
        # Calculate Gini coefficient
        def gini_coefficient(x):
            sorted_x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(sorted_x)
            return (2 * np.sum((np.arange(1, n+1) * sorted_x))) / (n * cumsum[-1]) - (n + 1) / n
        
        results['user_gini_coefficient'] = gini_coefficient(user_counts.values)
        results['top_20_percent_contribution'] = user_counts.head(int(len(user_counts) * 0.2)).sum() / user_counts.sum()
        
        # Q3.2: Author association analysis
        association_stats = self.df.groupby('author_association').agg({
            'is_closed': 'mean',
            'resolution_hours': 'mean',
            'comments': 'mean',
            'label_count': 'mean'
        }).to_dict('index')
        results['author_association_stats'] = association_stats
        
        # Q3.3: User evolution (for top users)
        user_evolution = {}
        for user in user_counts.head(10).index:
            user_issues = self.df[self.df['user_login'] == user].sort_values('created_at')
            if len(user_issues) >= 5:
                # Split into early and late issues
                mid_point = len(user_issues) // 2
                early_stats = {
                    'avg_comments': user_issues.iloc[:mid_point]['comments'].mean(),
                    'resolution_rate': user_issues.iloc[:mid_point]['is_closed'].mean()
                }
                late_stats = {
                    'avg_comments': user_issues.iloc[mid_point:]['comments'].mean(),
                    'resolution_rate': user_issues.iloc[mid_point:]['is_closed'].mean()
                }
                user_evolution[user] = {'early': early_stats, 'late': late_stats}
        
        results['user_evolution'] = user_evolution
        self.results['users'] = results
        
    def complexity_analysis(self):
        """Q4: Issue Complexity Metrics"""
        print(f"\n[{self.framework}] Running complexity analysis...")
        results = {}
        
        # Q4.1: Discussion intensity
        comment_stats = {
            'mean': self.df['comments'].mean(),
            'median': self.df['comments'].median(),
            'p90': self.df['comments'].quantile(0.9),
            'p99': self.df['comments'].quantile(0.99),
            'zero_comment_rate': (self.df['comments'] == 0).mean()
        }
        results['comment_distribution'] = comment_stats
        
        # High discussion issues
        high_discussion = self.df[self.df['comments'] >= self.df['comments'].quantile(0.9)]
        results['high_discussion_resolution_rate'] = high_discussion['is_closed'].mean()
        
        # Q4.2: Reaction patterns
        reaction_counts = []
        for _, row in self.df.iterrows():
            if isinstance(row['reactions'], dict):
                total_reactions = sum(v for k, v in row['reactions'].items() if k != 'url' and isinstance(v, int))
                reaction_counts.append(total_reactions)
            else:
                reaction_counts.append(0)
        
        self.df['total_reactions'] = reaction_counts
        results['reaction_stats'] = {
            'mean': np.mean(reaction_counts),
            'median': np.median(reaction_counts),
            'max': max(reaction_counts),
            'zero_reaction_rate': (np.array(reaction_counts) == 0).mean()
        }
        
        # Q4.3: State reason analysis
        state_reason_counts = self.df['state_reason'].value_counts().to_dict()
        results['state_reasons'] = state_reason_counts
        results['completion_rate'] = (self.df['state_reason'] == 'completed').sum() / len(self.df[self.df['is_closed']])
        
        self.results['complexity'] = results
        
    def anomaly_detection(self):
        """Q8: Statistical Anomaly Detection"""
        print(f"\n[{self.framework}] Running anomaly detection...")
        results = {}
        
        # Calculate z-scores for numeric fields
        numeric_fields = ['comments', 'total_reactions', 'resolution_hours', 'label_count']
        anomalies = {}
        
        for field in numeric_fields:
            if field in self.df.columns:
                data = self.df[field].dropna()
                if len(data) > 0:
                    mean = data.mean()
                    std = data.std()
                    if std > 0:
                        z_scores = np.abs((data - mean) / std)
                        outliers = self.df.loc[data[z_scores > 3].index]
                        anomalies[field] = {
                            'count': len(outliers),
                            'percentage': len(outliers) / len(data) * 100,
                            'examples': outliers[['number', 'title', field]].head(5).to_dict('records')
                        }
        
        results['field_anomalies'] = anomalies
        
        # Temporal anomalies (spikes in activity)
        daily_counts = self.df.groupby(self.df['created_at'].dt.date).size()
        if len(daily_counts) > 30:
            rolling_mean = daily_counts.rolling(window=7, center=True).mean()
            rolling_std = daily_counts.rolling(window=7, center=True).std()
            spikes = daily_counts[daily_counts > rolling_mean + 2 * rolling_std]
            results['activity_spikes'] = {
                'count': len(spikes),
                'dates': spikes.index.tolist()[:10],
                'max_spike_multiplier': (spikes / rolling_mean).max() if len(spikes) > 0 else 0
            }
        
        self.results['anomalies'] = results
        
    def comparative_metrics(self):
        """Calculate metrics for cross-framework comparison"""
        print(f"\n[{self.framework}] Calculating comparative metrics...")
        metrics = {}
        
        # Maturity indicators
        metrics['open_closed_ratio'] = (~self.df['is_closed']).sum() / self.df['is_closed'].sum()
        metrics['avg_resolution_hours'] = self.df['resolution_hours'].mean()
        metrics['label_entropy'] = -sum(p * np.log(p) for p in 
                                       self.df['label_count'].value_counts(normalize=True) if p > 0)
        
        # Community health
        metrics['unique_users'] = self.df['user_login'].nunique()
        metrics['comments_per_issue'] = self.df['comments'].mean()
        metrics['avg_response_hours'] = self.results.get('temporal', {}).get('response_time_stats', {}).get('mean_hours', None)
        
        # Activity metrics
        recent_30d = self.df[self.df['created_at'] >= (self.df['created_at'].max() - pd.Timedelta(days=30))]
        metrics['recent_activity'] = len(recent_30d)
        metrics['recent_unique_users'] = recent_30d['user_login'].nunique()
        
        self.results['comparative_metrics'] = metrics
        
    def generate_visualizations(self, output_dir):
        """Generate visualization plots"""
        print(f"\n[{self.framework}] Generating visualizations...")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Issue volume over time
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_counts = self.df.groupby('created_month').size()
        monthly_counts.plot(ax=ax, kind='line', marker='o')
        ax.set_title(f'{self.framework} - Monthly Issue Volume')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Issues')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.framework}_monthly_volume.png')
        plt.close()
        
        # 2. Resolution time distribution
        if 'resolution_hours' in self.df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            resolution_data = self.df['resolution_hours'].dropna()
            resolution_data[resolution_data < resolution_data.quantile(0.95)].hist(bins=50, ax=ax)
            ax.set_title(f'{self.framework} - Issue Resolution Time Distribution (95th percentile)')
            ax.set_xlabel('Hours to Resolution')
            ax.set_ylabel('Count')
            plt.tight_layout()
            plt.savefig(output_dir / f'{self.framework}_resolution_time.png')
            plt.close()
        
        # 3. Activity heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Hour of day
        hourly_counts = self.df['created_hour'].value_counts().sort_index()
        ax1.bar(hourly_counts.index, hourly_counts.values)
        ax1.set_title('Issues by Hour of Day')
        ax1.set_xlabel('Hour (UTC)')
        ax1.set_ylabel('Count')
        
        # Day of week
        dow_counts = self.df['created_dow'].value_counts().sort_index()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax2.bar(dow_counts.index, dow_counts.values)
        ax2.set_xticklabels(dow_names)
        ax2.set_title('Issues by Day of Week')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Count')
        
        plt.suptitle(f'{self.framework} - Activity Patterns')
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.framework}_activity_patterns.png')
        plt.close()
        
        # 4. User contribution distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        user_counts = self.df['user_login'].value_counts()
        top_users = user_counts.head(20)
        ax.barh(range(len(top_users)), top_users.values)
        ax.set_yticks(range(len(top_users)))
        ax.set_yticklabels(top_users.index)
        ax.set_xlabel('Number of Issues')
        ax.set_title(f'{self.framework} - Top 20 Issue Contributors')
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.framework}_top_contributors.png')
        plt.close()
        
    def generate_report(self, output_path):
        """Generate comprehensive report"""
        print(f"\n[{self.framework}] Generating report...")
        
        report = f"""# {self.framework} Issue Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

- Total Issues: {len(self.df)}
- Open Issues: {(~self.df['is_closed']).sum()}
- Closed Issues: {self.df['is_closed'].sum()}
- Date Range: {self.df['created_at'].min()} to {self.df['created_at'].max()}

## Temporal Analysis

### Issue Volume Trends
- Average issues per month: {self.results['temporal']['avg_issues_per_month']:.1f}
- Trend direction: {self.results['temporal']['trend_direction']}
- Peak activity hours (UTC): {self.results['temporal']['peak_hours']}
- Peak activity days: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][i] for i in self.results['temporal']['peak_days']]}

### Resolution Times
"""
        if 'resolution_stats' in self.results['temporal']:
            stats = self.results['temporal']['resolution_stats']
            report += f"""- Mean: {stats['mean_hours']:.1f} hours ({stats['mean_hours']/24:.1f} days)
- Median: {stats['median_hours']:.1f} hours ({stats['median_hours']/24:.1f} days)
- 90th percentile: {stats['p90_hours']:.1f} hours ({stats['p90_hours']/24:.1f} days)
- 99th percentile: {stats['p99_hours']:.1f} hours ({stats['p99_hours']/24:.1f} days)
"""

        report += f"""
## Label Analysis

- Total unique labels: {self.results['labels']['total_unique_labels']}
- Top 10 labels:
"""
        for label, count in list(self.results['labels']['top_labels'].items())[:10]:
            report += f"  - {label}: {count}\n"

        report += f"""
## User Analysis

- Total unique users: {self.results['users']['total_unique_users']}
- User contribution inequality (Gini): {self.results['users']['user_gini_coefficient']:.3f}
- Top 20% users contribute: {self.results['users']['top_20_percent_contribution']*100:.1f}% of issues

### Top 5 Contributors:
"""
        for user, count in list(self.results['users']['top_contributors'].items())[:5]:
            report += f"  - {user}: {count} issues\n"

        report += f"""
## Complexity Metrics

### Comment Distribution
- Mean comments per issue: {self.results['complexity']['comment_distribution']['mean']:.2f}
- Median comments: {self.results['complexity']['comment_distribution']['median']:.0f}
- Zero comment rate: {self.results['complexity']['comment_distribution']['zero_comment_rate']*100:.1f}%

### State Reasons
"""
        for reason, count in self.results['complexity']['state_reasons'].items():
            report += f"  - {reason}: {count}\n"

        report += f"""
## Anomaly Detection

### Field Anomalies (z-score > 3)
"""
        for field, data in self.results['anomalies']['field_anomalies'].items():
            report += f"- {field}: {data['count']} outliers ({data['percentage']:.2f}%)\n"

        # Write report
        with open(output_path, 'w') as f:
            f.write(report)
        
        # Also save raw results as JSON
        json_path = str(output_path).replace('.md', '_data.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
    def run_all_analyses(self):
        """Run all analysis methods"""
        self.load_data()
        self.temporal_analysis()
        self.label_analysis()
        self.user_analysis()
        self.complexity_analysis()
        self.anomaly_detection()
        self.comparative_metrics()


def compare_frameworks(analyzers):
    """Generate cross-framework comparison"""
    print("\nGenerating framework comparison...")
    
    comparison = {}
    for name, analyzer in analyzers.items():
        metrics = analyzer.results.get('comparative_metrics', {})
        comparison[name] = metrics
    
    # Create comparison DataFrame
    df_compare = pd.DataFrame(comparison).T
    
    # Generate comparison report
    report = """# Cross-Framework Comparison

## Key Metrics Comparison

| Metric | vLLM | SGLang | llama.cpp |
|--------|------|--------|-----------|
"""
    
    for metric in df_compare.columns:
        row = "| " + metric.replace('_', ' ').title() + " | "
        for framework in ['vllm', 'sglang', 'llama_cpp']:
            if framework in df_compare.index:
                value = df_compare.loc[framework, metric]
                if pd.notna(value):
                    if isinstance(value, float):
                        row += f"{value:.2f} | "
                    else:
                        row += f"{value} | "
                else:
                    row += "N/A | "
        report += row + "\n"
    
    # Add insights
    report += """
## Key Insights

### Maturity Comparison
"""
    
    # Find framework with best metrics
    best_resolution = df_compare['avg_resolution_hours'].idxmin()
    most_users = df_compare['unique_users'].idxmax()
    most_active = df_compare['recent_activity'].idxmax()
    
    report += f"""- Fastest average resolution: {best_resolution}
- Largest community: {most_users} ({df_compare.loc[most_users, 'unique_users']:.0f} unique users)
- Most recent activity: {most_active} ({df_compare.loc[most_active, 'recent_activity']:.0f} issues in last 30 days)
"""
    
    return report


def main():
    """Main analysis pipeline"""
    base_dir = Path("/root/yunwei37/vllm-exp/bug-study")
    data_dir = base_dir / "data"
    output_dir = base_dir / "analysis" / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Define datasets
    datasets = {
        'vllm': data_dir / 'vllm_all_issues.json',
        'sglang': data_dir / 'sglang_issues.json',
        'llama_cpp': data_dir / 'llama_cpp_issues.json'
    }
    
    # Run analysis for each framework
    analyzers = {}
    for name, path in datasets.items():
        if path.exists():
            print(f"\n{'='*60}")
            print(f"Analyzing {name}")
            print(f"{'='*60}")
            
            analyzer = IssueAnalyzer(name, path)
            analyzer.run_all_analyses()
            analyzer.generate_visualizations(output_dir / 'plots')
            analyzer.generate_report(output_dir / f'{name}_analysis_report.md')
            analyzers[name] = analyzer
        else:
            print(f"Warning: {path} not found, skipping {name}")
    
    # Generate comparison report
    if analyzers:
        comparison_report = compare_frameworks(analyzers)
        with open(output_dir / 'framework_comparison.md', 'w') as f:
            f.write(comparison_report)
        
        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()