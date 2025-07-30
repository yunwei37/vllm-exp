#!/usr/bin/env python3
"""
Long-Tail Issues Analysis: Analyze issues with high comment counts to understand
engagement patterns, complexity indicators, and resolution challenges
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class LongTailIssuesAnalyzer:
    def __init__(self, framework_name, data_path, output_dir):
        self.framework = framework_name
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) / framework_name.lower().replace('.', '_')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.long_tail_df = None
        self.results = {}
        
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
        self.df['updated_at'] = pd.to_datetime(self.df['updated_at'], errors='coerce')
        
        # Extract label names
        self.df['label_names'] = self.df['labels'].apply(
            lambda x: [l['name'] for l in x if isinstance(l, dict) and 'name' in l] 
            if isinstance(x, list) else []
        )
        
        # Basic metrics
        self.df['body_length'] = self.df['body'].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        self.df['title_length'] = self.df['title'].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        self.df['comments'] = pd.to_numeric(self.df['comments'], errors='coerce').fillna(0)
        self.df['is_closed'] = self.df['state'] == 'closed'
        
        # Calculate resolution time
        mask = self.df['is_closed'] & self.df['closed_at'].notna()
        self.df['resolution_days'] = pd.NaT
        self.df.loc[mask, 'resolution_days'] = (
            (self.df.loc[mask, 'closed_at'] - self.df.loc[mask, 'created_at']).dt.total_seconds() / 86400
        )
        
        # Calculate time since last update
        self.df['days_since_update'] = (
            (datetime.now(tz=self.df['updated_at'].dt.tz) - self.df['updated_at']).dt.total_seconds() / 86400
        )
        
        # Identify long-tail issues (high comment count)
        comment_threshold = self.df['comments'].quantile(0.9)  # Top 10% by comment count
        self.long_tail_df = self.df[self.df['comments'] >= comment_threshold].copy()
        print(f"Identified {len(self.long_tail_df)} long-tail issues (>= {comment_threshold:.0f} comments)")
        
    def analyze_comment_distribution(self):
        """Analyze comment count distribution and identify thresholds"""
        print(f"\n[{self.framework}] Analyzing comment distribution...")
        
        # Calculate distribution statistics
        comment_stats = {
            'mean': self.df['comments'].mean(),
            'median': self.df['comments'].median(),
            'std': self.df['comments'].std(),
            'p75': self.df['comments'].quantile(0.75),
            'p90': self.df['comments'].quantile(0.90),
            'p95': self.df['comments'].quantile(0.95),
            'p99': self.df['comments'].quantile(0.99),
            'max': self.df['comments'].max()
        }
        
        # Define long-tail categories
        self.df['comment_category'] = pd.cut(
            self.df['comments'],
            bins=[-1, comment_stats['median'], comment_stats['p75'], 
                  comment_stats['p90'], comment_stats['p95'], float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High', 'Extreme']
        )
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.framework} - Comment Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Comment distribution histogram
        ax = axes[0, 0]
        counts, bins, _ = ax.hist(self.df['comments'], bins=50, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.axvline(comment_stats['p90'], color='red', linestyle='--', linewidth=2, label=f"P90: {comment_stats['p90']:.0f}")
        ax.axvline(comment_stats['p95'], color='orange', linestyle='--', linewidth=2, label=f"P95: {comment_stats['p95']:.0f}")
        ax.set_xlabel('Number of Comments')
        ax.set_ylabel('Number of Issues')
        ax.set_title('Comment Count Distribution')
        ax.legend()
        ax.set_yscale('log')
        
        # 2. Box plot by state
        ax = axes[0, 1]
        self.df.boxplot(column='comments', by='state', ax=ax)
        ax.set_ylabel('Number of Comments')
        ax.set_title('Comment Distribution by Issue State')
        
        # 3. Comment categories
        ax = axes[1, 0]
        category_counts = self.df['comment_category'].value_counts()
        bars = ax.bar(range(len(category_counts)), category_counts.values, 
                      color=['lightgreen', 'yellow', 'orange', 'red', 'darkred'][:len(category_counts)])
        ax.set_xticks(range(len(category_counts)))
        ax.set_xticklabels(category_counts.index, rotation=45)
        ax.set_ylabel('Number of Issues')
        ax.set_title('Issues by Comment Category')
        
        # Add value labels
        for i, v in enumerate(category_counts.values):
            ax.text(i, v + 10, f"{v}\n({v/len(self.df)*100:.1f}%)", 
                   ha='center', fontsize=10, fontweight='bold')
        
        # 4. Cumulative distribution
        ax = axes[1, 1]
        sorted_comments = self.df['comments'].sort_values()
        cumulative = np.arange(1, len(sorted_comments) + 1) / len(sorted_comments) * 100
        ax.plot(sorted_comments, cumulative, linewidth=2, color='darkblue')
        ax.axhline(90, color='red', linestyle='--', alpha=0.5)
        ax.axhline(95, color='orange', linestyle='--', alpha=0.5)
        ax.set_xlabel('Number of Comments')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Cumulative Distribution of Comments')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework.lower()}_comment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.results['comment_stats'] = comment_stats
        return comment_stats
    
    def analyze_long_tail_characteristics(self):
        """Analyze characteristics of long-tail issues"""
        print(f"\n[{self.framework}] Analyzing long-tail issue characteristics...")
        
        if len(self.long_tail_df) == 0:
            print("No long-tail issues found!")
            return
        
        # Compare with regular issues
        regular_df = self.df[self.df['comments'] < self.df['comments'].quantile(0.9)]
        
        comparisons = {
            'long_tail_count': len(self.long_tail_df),
            'regular_count': len(regular_df),
            'avg_body_length': {
                'long_tail': self.long_tail_df['body_length'].mean(),
                'regular': regular_df['body_length'].mean()
            },
            'resolution_rate': {
                'long_tail': self.long_tail_df['is_closed'].mean() * 100,
                'regular': regular_df['is_closed'].mean() * 100
            },
            'avg_resolution_days': {
                'long_tail': self.long_tail_df['resolution_days'].mean(),
                'regular': regular_df['resolution_days'].mean()
            },
            'avg_labels': {
                'long_tail': self.long_tail_df['label_names'].apply(len).mean(),
                'regular': regular_df['label_names'].apply(len).mean()
            }
        }
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.framework} - Long-Tail vs Regular Issues Comparison', fontsize=16, fontweight='bold')
        
        # 1. Body length comparison
        ax = axes[0, 0]
        data = [regular_df['body_length'].dropna(), self.long_tail_df['body_length'].dropna()]
        bp = ax.boxplot(data, labels=['Regular', 'Long-tail'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Body Length (characters)')
        ax.set_title('Issue Body Length Comparison')
        ax.set_yscale('log')
        
        # 2. Resolution rate
        ax = axes[0, 1]
        categories = ['Regular', 'Long-tail']
        resolution_rates = [comparisons['resolution_rate']['regular'], 
                           comparisons['resolution_rate']['long_tail']]
        bars = ax.bar(categories, resolution_rates, color=['lightblue', 'lightcoral'])
        ax.set_ylabel('Resolution Rate (%)')
        ax.set_title('Issue Resolution Rate')
        for i, v in enumerate(resolution_rates):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # 3. Resolution time
        ax = axes[0, 2]
        regular_res = regular_df['resolution_days'].dropna()
        longtail_res = self.long_tail_df['resolution_days'].dropna()
        if len(regular_res) > 0 and len(longtail_res) > 0:
            data = [regular_res, longtail_res]
            bp = ax.boxplot(data, labels=['Regular', 'Long-tail'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            ax.set_ylabel('Resolution Time (days)')
            ax.set_title('Resolution Time Comparison')
            ax.set_yscale('log')
        
        # 4. Label distribution
        ax = axes[1, 0]
        regular_labels = Counter([label for labels in regular_df['label_names'] for label in labels])
        longtail_labels = Counter([label for labels in self.long_tail_df['label_names'] for label in labels])
        
        # Get top 10 labels from each
        top_regular = dict(regular_labels.most_common(10))
        top_longtail = dict(longtail_labels.most_common(10))
        all_labels = sorted(set(list(top_regular.keys()) + list(top_longtail.keys())))[:15]
        
        x = np.arange(len(all_labels))
        width = 0.35
        
        regular_counts = [regular_labels.get(label, 0) / len(regular_df) * 100 for label in all_labels]
        longtail_counts = [longtail_labels.get(label, 0) / len(self.long_tail_df) * 100 for label in all_labels]
        
        ax.bar(x - width/2, regular_counts, width, label='Regular', color='lightblue')
        ax.bar(x + width/2, longtail_counts, width, label='Long-tail', color='lightcoral')
        
        ax.set_xlabel('Labels')
        ax.set_ylabel('Percentage of Issues (%)')
        ax.set_title('Label Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        ax.legend()
        
        # 5. Time patterns
        ax = axes[1, 1]
        regular_dow = regular_df['created_at'].dt.dayofweek.value_counts().sort_index()
        longtail_dow = self.long_tail_df['created_at'].dt.dayofweek.value_counts().sort_index()
        
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        x = np.arange(len(days))
        
        regular_norm = regular_dow / regular_dow.sum() * 100
        longtail_norm = longtail_dow / longtail_dow.sum() * 100
        
        ax.plot(x, regular_norm, marker='o', label='Regular', linewidth=2, markersize=8)
        ax.plot(x, longtail_norm, marker='s', label='Long-tail', linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(days)
        ax.set_ylabel('Percentage of Issues (%)')
        ax.set_title('Issue Creation by Day of Week')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. User engagement metrics
        ax = axes[1, 2]
        metrics = ['Avg Comments', 'Avg Body Length', 'Avg Labels', 'Resolution Rate']
        regular_vals = [
            regular_df['comments'].mean(),
            regular_df['body_length'].mean() / 1000,  # Scale down for visualization
            regular_df['label_names'].apply(len).mean() * 10,  # Scale up for visualization
            comparisons['resolution_rate']['regular']
        ]
        longtail_vals = [
            self.long_tail_df['comments'].mean(),
            self.long_tail_df['body_length'].mean() / 1000,
            self.long_tail_df['label_names'].apply(len).mean() * 10,
            comparisons['resolution_rate']['long_tail']
        ]
        
        x = np.arange(len(metrics))
        ax.bar(x - width/2, regular_vals, width, label='Regular', color='lightblue')
        ax.bar(x + width/2, longtail_vals, width, label='Long-tail', color='lightcoral')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('Value (normalized)')
        ax.set_title('Engagement Metrics Comparison')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework.lower()}_longtail_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.results['comparisons'] = comparisons
        return comparisons
    
    def analyze_label_distribution_longtail(self):
        """Analyze and visualize label distribution in long-tail issues"""
        print(f"\n[{self.framework}] Analyzing label distribution in long-tail issues...")
        
        # Count all labels in long-tail issues
        label_counts = Counter()
        for labels in self.long_tail_df['label_names']:
            label_counts.update(labels)
        
        if not label_counts:
            print("No labels found in long-tail issues!")
            return {}
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'{self.framework} - Label Distribution in Long-Tail Issues', fontsize=16, fontweight='bold')
        
        # 1. All labels in long-tail issues (horizontal bar chart)
        ax = axes[0, 0]
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Adjust figure height if needed
        if len(sorted_labels) > 20:
            labels_to_show = sorted_labels[:20]
            title_suffix = " (Top 20)"
        else:
            labels_to_show = sorted_labels
            title_suffix = " (All)"
        
        labels, counts = zip(*labels_to_show)
        y_pos = np.arange(len(labels))
        
        bars = ax.barh(y_pos, counts, color='darkblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Number of Long-Tail Issues')
        ax.set_title(f'Label Frequency in Long-Tail Issues{title_suffix}')
        
        # Add value labels
        for i, v in enumerate(counts):
            percentage = v / len(self.long_tail_df) * 100
            ax.text(v + 1, i, f'{v} ({percentage:.1f}%)', va='center', fontsize=9, fontweight='bold')
        
        # 2. Comparison: Label usage in long-tail vs regular issues
        ax = axes[0, 1]
        regular_df = self.df[self.df['comments'] < self.df['comments'].quantile(0.9)]
        
        # Count labels in regular issues
        regular_label_counts = Counter()
        for labels in regular_df['label_names']:
            regular_label_counts.update(labels)
        
        # Get top 15 labels from long-tail
        top_labels = [label for label, _ in sorted_labels[:15]]
        
        # Calculate percentages
        longtail_pcts = [label_counts[label] / len(self.long_tail_df) * 100 for label in top_labels]
        regular_pcts = [regular_label_counts.get(label, 0) / len(regular_df) * 100 for label in top_labels]
        
        x = np.arange(len(top_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, regular_pcts, width, label='Regular Issues', color='lightblue')
        bars2 = ax.bar(x + width/2, longtail_pcts, width, label='Long-tail Issues', color='darkred')
        
        ax.set_xlabel('Labels')
        ax.set_ylabel('Percentage of Issues (%)')
        ax.set_title('Label Usage: Long-tail vs Regular Issues')
        ax.set_xticks(x)
        ax.set_xticklabels(top_labels, rotation=45, ha='right', fontsize=9)
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.5:  # Only show label if bar is visible
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Label combinations in long-tail issues
        ax = axes[1, 0]
        # Count issues by number of labels
        label_count_dist = self.long_tail_df['label_names'].apply(len).value_counts().sort_index()
        
        bars = ax.bar(label_count_dist.index, label_count_dist.values, color='green', alpha=0.7)
        ax.set_xlabel('Number of Labels per Issue')
        ax.set_ylabel('Number of Long-Tail Issues')
        ax.set_title('Label Count Distribution in Long-Tail Issues')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Top label combinations
        ax = axes[1, 1]
        # Find most common label combinations
        label_combinations = []
        for labels in self.long_tail_df['label_names']:
            if len(labels) > 1:
                # Sort labels to ensure consistent combinations
                combo = ', '.join(sorted(labels))
                label_combinations.append(combo)
        
        if label_combinations:
            combo_counts = Counter(label_combinations).most_common(10)
            if combo_counts:
                combos, counts = zip(*combo_counts)
                y_pos = np.arange(len(combos))
                
                bars = ax.barh(y_pos, counts, color='purple', alpha=0.7)
                ax.set_yticks(y_pos)
                # Truncate long combinations for display
                ax.set_yticklabels([combo[:40] + '...' if len(combo) > 40 else combo 
                                   for combo in combos], fontsize=8)
                ax.set_xlabel('Frequency')
                ax.set_title('Top Label Combinations in Long-Tail Issues')
                
                # Add value labels
                for i, v in enumerate(counts):
                    ax.text(v + 0.1, i, str(v), va='center', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No multi-label issues found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Top Label Combinations in Long-Tail Issues')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework.lower()}_label_distribution_longtail.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['label_distribution'] = {
            'total_labels': len(label_counts),
            'most_common': sorted_labels[:10],
            'label_count_distribution': label_count_dist.to_dict()
        }
        
        return label_counts
    
    def analyze_top_long_tail_issues(self):
        """Analyze top long-tail issues in detail"""
        print(f"\n[{self.framework}] Analyzing top long-tail issues...")
        
        # Get top 20 issues by comment count
        top_issues = self.long_tail_df.nlargest(20, 'comments')
        
        # Create detailed analysis
        top_issues_analysis = []
        for _, issue in top_issues.iterrows():
            analysis = {
                'number': issue['number'],
                'title': issue['title'][:100] + '...' if len(issue['title']) > 100 else issue['title'],
                'comments': int(issue['comments']),
                'state': issue['state'],
                'labels': ', '.join(issue['label_names']),
                'created_days_ago': (datetime.now(tz=issue['created_at'].tz) - issue['created_at']).days,
                'body_length': int(issue['body_length']),
                'url': issue.get('html_url', '')
            }
            
            if issue['is_closed'] and pd.notna(issue['resolution_days']):
                analysis['resolution_days'] = round(issue['resolution_days'], 1)
            else:
                analysis['resolution_days'] = 'N/A'
                
            top_issues_analysis.append(analysis)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.framework} - Top Long-Tail Issues Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top issues by comments
        ax = axes[0, 0]
        top_10 = top_issues.head(10)
        y_pos = np.arange(len(top_10))
        bars = ax.barh(y_pos, top_10['comments'], color='darkred', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"#{row['number']}" for _, row in top_10.iterrows()], fontsize=9)
        ax.set_xlabel('Number of Comments')
        ax.set_title('Top 10 Issues by Comment Count')
        
        # Add value labels
        for i, v in enumerate(top_10['comments']):
            ax.text(v + 1, i, f'{int(v)}', va='center', fontweight='bold')
        
        # 2. State distribution of top issues
        ax = axes[0, 1]
        state_counts = top_issues['state'].value_counts()
        colors = ['green' if state == 'closed' else 'red' for state in state_counts.index]
        ax.pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax.set_title('State Distribution of Top Long-Tail Issues')
        
        # 3. Label frequency in top issues
        ax = axes[1, 0]
        all_labels = [label for labels in top_issues['label_names'] for label in labels]
        label_counts = Counter(all_labels).most_common(10)
        if label_counts:
            labels, counts = zip(*label_counts)
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, counts, color='purple', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel('Frequency')
            ax.set_title('Most Common Labels in Top Long-Tail Issues')
            
            # Add value labels
            for i, v in enumerate(counts):
                ax.text(v + 0.1, i, str(v), va='center', fontweight='bold')
        
        # 4. Age distribution
        ax = axes[1, 1]
        ages = [(datetime.now(tz=issue['created_at'].tz) - issue['created_at']).days 
                for _, issue in top_issues.iterrows()]
        ax.hist(ages, bins=10, color='orange', edgecolor='darkorange', alpha=0.7)
        ax.set_xlabel('Age (days)')
        ax.set_ylabel('Number of Issues')
        ax.set_title('Age Distribution of Top Long-Tail Issues')
        ax.axvline(np.mean(ages), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(ages):.0f} days')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework.lower()}_top_longtail_issues.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.results['top_issues'] = top_issues_analysis
        return top_issues_analysis
    
    def analyze_user_patterns(self):
        """Analyze user interaction patterns in long-tail issues"""
        print(f"\n[{self.framework}] Analyzing user patterns in long-tail issues...")
        
        # Analyze user (author) patterns
        regular_df = self.df[self.df['comments'] < self.df['comments'].quantile(0.9)]
        
        # Count unique authors
        regular_authors = regular_df['user'].apply(lambda x: x['login'] if isinstance(x, dict) else None).dropna()
        longtail_authors = self.long_tail_df['user'].apply(lambda x: x['login'] if isinstance(x, dict) else None).dropna()
        
        author_stats = {
            'unique_authors_regular': regular_authors.nunique(),
            'unique_authors_longtail': longtail_authors.nunique(),
            'avg_issues_per_author_regular': len(regular_df) / regular_authors.nunique(),
            'avg_issues_per_author_longtail': len(self.long_tail_df) / longtail_authors.nunique()
        }
        
        # Top contributors to long-tail issues
        longtail_author_counts = longtail_authors.value_counts()
        top_longtail_authors = longtail_author_counts.head(15)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.framework} - User Patterns in Long-Tail Issues', fontsize=16, fontweight='bold')
        
        # 1. Top authors of long-tail issues
        ax = axes[0, 0]
        y_pos = np.arange(len(top_longtail_authors))
        ax.barh(y_pos, top_longtail_authors.values, color='teal', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_longtail_authors.index, fontsize=9)
        ax.set_xlabel('Number of Long-Tail Issues Created')
        ax.set_title('Top Authors of Long-Tail Issues')
        
        # Add value labels
        for i, v in enumerate(top_longtail_authors.values):
            ax.text(v + 0.1, i, str(v), va='center', fontweight='bold')
        
        # 2. Author overlap analysis
        ax = axes[0, 1]
        regular_author_set = set(regular_authors)
        longtail_author_set = set(longtail_authors)
        
        overlap = len(regular_author_set & longtail_author_set)
        only_regular = len(regular_author_set - longtail_author_set)
        only_longtail = len(longtail_author_set - regular_author_set)
        
        venn_data = [only_regular, overlap, only_longtail]
        labels = ['Only Regular', 'Both', 'Only Long-tail']
        colors = ['lightblue', 'purple', 'lightcoral']
        
        ax.pie(venn_data, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Author Distribution Across Issue Types')
        
        # 3. Engagement patterns over time
        ax = axes[1, 0]
        # Group by month and calculate average comments
        self.long_tail_df['month'] = self.long_tail_df['created_at'].dt.to_period('M')
        monthly_avg = self.long_tail_df.groupby('month')['comments'].agg(['mean', 'count'])
        
        if len(monthly_avg) > 0:
            months = monthly_avg.index.astype(str)
            ax.plot(range(len(months)), monthly_avg['mean'], marker='o', linewidth=2, 
                   markersize=8, label='Avg Comments')
            ax2 = ax.twinx()
            ax2.bar(range(len(months)), monthly_avg['count'], alpha=0.3, color='gray', 
                   label='Issue Count')
            
            ax.set_xticks(range(0, len(months), max(1, len(months)//10)))
            ax.set_xticklabels(months[::max(1, len(months)//10)], rotation=45, ha='right')
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Comments', color='blue')
            ax2.set_ylabel('Number of Long-Tail Issues', color='gray')
            ax.set_title('Long-Tail Issue Trends Over Time')
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='gray')
        
        # 4. Label correlation with engagement
        ax = axes[1, 1]
        # Calculate average comments per label
        label_engagement = defaultdict(list)
        for _, issue in self.long_tail_df.iterrows():
            for label in issue['label_names']:
                label_engagement[label].append(issue['comments'])
        
        label_avg_comments = {label: np.mean(comments) for label, comments in label_engagement.items() 
                             if len(comments) >= 3}  # At least 3 issues
        
        if label_avg_comments:
            sorted_labels = sorted(label_avg_comments.items(), key=lambda x: x[1], reverse=True)[:10]
            labels, avg_comments = zip(*sorted_labels)
            
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, avg_comments, color='darkgreen', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel('Average Comments')
            ax.set_title('Labels with Highest Average Engagement')
            
            # Add value labels
            for i, v in enumerate(avg_comments):
                ax.text(v + 0.5, i, f'{v:.1f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework.lower()}_user_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.results['author_stats'] = author_stats
        return author_stats
    
    def generate_report(self):
        """Generate comprehensive report on long-tail issues"""
        print(f"\n[{self.framework}] Generating long-tail issues report...")
        
        report_path = self.output_dir / f'{self.framework.lower()}_longtail_report.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# {self.framework} - Long-Tail Issues Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Issues Analyzed**: {len(self.df)}\n")
            f.write(f"- **Long-Tail Issues** (≥P90 comments): {len(self.long_tail_df)}\n")
            f.write(f"- **Long-Tail Threshold**: {self.df['comments'].quantile(0.9):.0f} comments\n")
            f.write(f"- **Percentage of Long-Tail**: {len(self.long_tail_df)/len(self.df)*100:.1f}%\n\n")
            
            # Comment Distribution Stats
            f.write("## Comment Distribution Statistics\n\n")
            if 'comment_stats' in self.results:
                stats = self.results['comment_stats']
                f.write(f"- **Mean Comments**: {stats['mean']:.1f}\n")
                f.write(f"- **Median Comments**: {stats['median']:.0f}\n")
                f.write(f"- **P90 Comments**: {stats['p90']:.0f}\n")
                f.write(f"- **P95 Comments**: {stats['p95']:.0f}\n")
                f.write(f"- **P99 Comments**: {stats['p99']:.0f}\n")
                f.write(f"- **Max Comments**: {stats['max']:.0f}\n\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            if 'comparisons' in self.results:
                comp = self.results['comparisons']
                f.write("### Long-Tail vs Regular Issues\n\n")
                f.write(f"- **Average Body Length**:\n")
                f.write(f"  - Long-tail: {comp['avg_body_length']['long_tail']:.0f} characters\n")
                f.write(f"  - Regular: {comp['avg_body_length']['regular']:.0f} characters\n")
                f.write(f"  - Ratio: {comp['avg_body_length']['long_tail']/comp['avg_body_length']['regular']:.2f}x\n\n")
                
                f.write(f"- **Resolution Rate**:\n")
                f.write(f"  - Long-tail: {comp['resolution_rate']['long_tail']:.1f}%\n")
                f.write(f"  - Regular: {comp['resolution_rate']['regular']:.1f}%\n")
                f.write(f"  - Difference: {comp['resolution_rate']['long_tail']-comp['resolution_rate']['regular']:+.1f}%\n\n")
                
                if pd.notna(comp['avg_resolution_days']['long_tail']):
                    f.write(f"- **Average Resolution Time**:\n")
                    f.write(f"  - Long-tail: {comp['avg_resolution_days']['long_tail']:.1f} days\n")
                    f.write(f"  - Regular: {comp['avg_resolution_days']['regular']:.1f} days\n")
                    f.write(f"  - Ratio: {comp['avg_resolution_days']['long_tail']/comp['avg_resolution_days']['regular']:.2f}x\n\n")
            
            # Top Long-Tail Issues
            f.write("## Top 10 Long-Tail Issues\n\n")
            if 'top_issues' in self.results:
                f.write("| # | Title | Comments | State | Labels | Age (days) | Resolution |\n")
                f.write("|---|-------|----------|-------|--------|------------|------------|\n")
                for issue in self.results['top_issues'][:10]:
                    f.write(f"| {issue['number']} | {issue['title']} | {issue['comments']} | "
                           f"{issue['state']} | {issue['labels'] or 'None'} | {issue['created_days_ago']} | "
                           f"{issue['resolution_days']} |\n")
            
            # User Patterns
            f.write("\n## User Engagement Patterns\n\n")
            if 'author_stats' in self.results:
                stats = self.results['author_stats']
                f.write(f"- **Unique Authors (Regular Issues)**: {stats['unique_authors_regular']}\n")
                f.write(f"- **Unique Authors (Long-tail Issues)**: {stats['unique_authors_longtail']}\n")
                f.write(f"- **Avg Issues per Author (Regular)**: {stats['avg_issues_per_author_regular']:.2f}\n")
                f.write(f"- **Avg Issues per Author (Long-tail)**: {stats['avg_issues_per_author_longtail']:.2f}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the analysis, consider the following strategies for managing long-tail issues:\n\n")
            
            if len(self.long_tail_df) > 0:
                # Check resolution rate difference
                if 'comparisons' in self.results:
                    comp = self.results['comparisons']
                    if comp['resolution_rate']['long_tail'] < comp['resolution_rate']['regular']:
                        f.write("1. **Improve Resolution Rate**: Long-tail issues have lower resolution rates. "
                               "Consider dedicating more resources or creating specialized teams.\n\n")
                    
                    if comp['avg_body_length']['long_tail'] > comp['avg_body_length']['regular'] * 1.5:
                        f.write("2. **Issue Decomposition**: Long-tail issues tend to be more complex. "
                               "Consider breaking them down into smaller, manageable tasks.\n\n")
                
                # Check label patterns
                longtail_labels = Counter([label for labels in self.long_tail_df['label_names'] for label in labels])
                if longtail_labels:
                    top_label = longtail_labels.most_common(1)[0]
                    f.write(f"3. **Label-Specific Strategies**: The '{top_label[0]}' label appears most frequently "
                           f"in long-tail issues. Consider specific workflows for this category.\n\n")
                
                # Check age patterns
                old_issues = self.long_tail_df[
                    (datetime.now(tz=self.long_tail_df['created_at'].dt.tz) - self.long_tail_df['created_at']).dt.days > 180
                ]
                if len(old_issues) > len(self.long_tail_df) * 0.3:
                    f.write("4. **Address Aging Issues**: Over 30% of long-tail issues are older than 6 months. "
                           "Consider periodic reviews and cleanup campaigns.\n\n")
            
            f.write("\n---\n")
            f.write(f"*Report generated by Long-Tail Issues Analyzer*\n")
        
        print(f"Report saved to: {report_path}")
        return report_path
    
    def save_data(self):
        """Save processed data for further analysis"""
        # Save long-tail issues data
        longtail_data = self.long_tail_df[['number', 'title', 'state', 'comments', 'body_length', 
                                           'label_names', 'created_at', 'closed_at', 'resolution_days']].to_dict('records')
        
        output_data = {
            'framework': self.framework,
            'analysis_date': datetime.now().isoformat(),
            'total_issues': len(self.df),
            'longtail_count': len(self.long_tail_df),
            'longtail_threshold': float(self.df['comments'].quantile(0.9)),
            'results': self.results,
            'longtail_issues': longtail_data
        }
        
        json_path = self.output_dir / f'{self.framework.lower()}_longtail_data.json'
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"Data saved to: {json_path}")


def analyze_all_frameworks():
    """Analyze long-tail issues for all frameworks"""
    frameworks = [
        ('vLLM', '../data/vllm_all_issues.json'),
        ('SGLang', '../data/sglang_issues.json'),
        ('llama.cpp', '../data/llama_cpp_issues.json')
    ]
    
    output_base = Path('results/long_tail_analysis')
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for framework, data_path in frameworks:
        print(f"\n{'='*80}")
        print(f"Analyzing {framework}")
        print(f"{'='*80}")
        
        try:
            analyzer = LongTailIssuesAnalyzer(framework, data_path, output_base)
            analyzer.load_data()
            
            # Run all analyses
            analyzer.analyze_comment_distribution()
            analyzer.analyze_long_tail_characteristics()
            analyzer.analyze_label_distribution_longtail()
            analyzer.analyze_top_long_tail_issues()
            analyzer.analyze_user_patterns()
            
            # Generate report and save data
            analyzer.generate_report()
            analyzer.save_data()
            
            all_results[framework] = analyzer.results
            
        except Exception as e:
            print(f"Error analyzing {framework}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comparative summary
    generate_comparative_summary(all_results, output_base)
    
    return all_results


def generate_comparative_summary(all_results, output_base):
    """Generate a comparative summary across all frameworks"""
    summary_path = output_base / 'comparative_summary.md'
    
    with open(summary_path, 'w') as f:
        f.write("# Comparative Long-Tail Issues Analysis\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Framework Comparison\n\n")
        f.write("| Framework | Total Issues | Long-tail Issues | Long-tail % | P90 Threshold | Max Comments |\n")
        f.write("|-----------|-------------|------------------|-------------|---------------|-------------|\n")
        
        for framework, results in all_results.items():
            if 'comment_stats' in results and 'comparisons' in results:
                stats = results['comment_stats']
                comp = results['comparisons']
                f.write(f"| {framework} | {comp['regular_count'] + comp['long_tail_count']} | "
                       f"{comp['long_tail_count']} | "
                       f"{comp['long_tail_count']/(comp['regular_count']+comp['long_tail_count'])*100:.1f}% | "
                       f"{stats['p90']:.0f} | {stats['max']:.0f} |\n")
        
        f.write("\n## Key Insights\n\n")
        
        # Find framework with highest long-tail percentage
        max_longtail_pct = 0
        max_longtail_framework = ""
        for framework, results in all_results.items():
            if 'comparisons' in results:
                comp = results['comparisons']
                pct = comp['long_tail_count'] / (comp['regular_count'] + comp['long_tail_count']) * 100
                if pct > max_longtail_pct:
                    max_longtail_pct = pct
                    max_longtail_framework = framework
        
        f.write(f"- **Highest Long-tail Percentage**: {max_longtail_framework} ({max_longtail_pct:.1f}%)\n")
        
        # Compare resolution rates
        f.write("\n### Resolution Rate Differences (Long-tail vs Regular)\n\n")
        for framework, results in all_results.items():
            if 'comparisons' in results:
                comp = results['comparisons']
                diff = comp['resolution_rate']['long_tail'] - comp['resolution_rate']['regular']
                f.write(f"- **{framework}**: {diff:+.1f}% ")
                if diff < -10:
                    f.write("⚠️ (Significant negative impact)\n")
                elif diff < 0:
                    f.write("(Moderate negative impact)\n")
                else:
                    f.write("✅ (No negative impact)\n")
        
        f.write("\n---\n")
        f.write("*Comparative analysis of long-tail issues across LLM serving frameworks*\n")
    
    print(f"\nComparative summary saved to: {summary_path}")


if __name__ == "__main__":
    results = analyze_all_frameworks()
    
    print("\n" + "="*80)
    print("Long-Tail Issues Analysis Complete!")
    print("="*80)
    print("\nResults have been saved to the 'results/long_tail_analysis' directory.")
    print("Each framework has:")
    print("  - Multiple visualization charts")
    print("  - Detailed markdown report")
    print("  - JSON data file for further analysis")
    print("\nA comparative summary has also been generated.")