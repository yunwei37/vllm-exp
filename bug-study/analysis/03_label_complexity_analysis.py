#!/usr/bin/env python3
"""
Label and Complexity Analysis Module: Analyze issue categorization and complexity indicators
Research Questions:
- Label usage patterns and evolution
- Issue complexity indicators
- Label effectiveness for issue resolution
- Cross-label correlations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class LabelComplexityAnalyzer:
    def __init__(self, framework_name, data_path, output_dir):
        self.framework = framework_name
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.results = {}
        
    def load_data(self):
        """Load and preprocess data"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        items = data.get('items', data if isinstance(data, list) else [])
        self.df = pd.DataFrame(items)
        
        # Convert timestamps
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
        self.df['closed_at'] = pd.to_datetime(self.df['closed_at'], errors='coerce')
        
        # Extract label information
        self.df['label_names'] = self.df['labels'].apply(
            lambda x: [l['name'] for l in x if isinstance(l, dict) and 'name' in l] if isinstance(x, list) else []
        )
        self.df['label_count'] = self.df['label_names'].apply(len)
        
        # Basic fields
        self.df['is_closed'] = self.df['state'] == 'closed'
        self.df['resolution_days'] = pd.NaT
        mask = self.df['is_closed'] & self.df['closed_at'].notna()
        self.df.loc[mask, 'resolution_days'] = (
            self.df.loc[mask, 'closed_at'] - self.df.loc[mask, 'created_at']
        ).dt.total_seconds() / 86400
        
        # Extract reactions
        self.df['total_reactions'] = self.df['reactions'].apply(
            lambda x: sum(v for k, v in x.items() if k != 'url' and isinstance(v, int)) if isinstance(x, dict) else 0
        )
        
    def analyze_label_distribution(self):
        """Analyze label usage patterns and distributions"""
        print(f"[{self.framework}] Analyzing label distribution...")
        
        # Collect all labels
        all_labels = []
        for labels in self.df['label_names']:
            all_labels.extend(labels)
        label_counts = Counter(all_labels)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.framework} - Label Distribution Analysis', fontsize=16, y=1.02)
        
        # 1. Top labels bar chart
        ax = axes[0, 0]
        top_labels = dict(label_counts.most_common(20))
        y_pos = np.arange(len(top_labels))
        ax.barh(y_pos, list(top_labels.values()), color='skyblue', edgecolor='navy', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(top_labels.keys()), fontsize=9)
        ax.set_xlabel('Number of Issues')
        ax.set_title('Top 20 Most Used Labels')
        ax.grid(True, axis='x', alpha=0.3)
        
        # 2. Label count distribution
        ax = axes[0, 1]
        label_count_dist = self.df['label_count'].value_counts().sort_index()
        ax.bar(label_count_dist.index, label_count_dist.values, color='coral', alpha=0.7)
        ax.set_xlabel('Number of Labels per Issue')
        ax.set_ylabel('Number of Issues')
        ax.set_title('Label Count Distribution')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add percentage annotations
        total_issues = len(self.df)
        for i, (count, freq) in enumerate(label_count_dist.items()):
            ax.text(count, freq + 50, f'{freq/total_issues*100:.1f}%', 
                   ha='center', fontsize=8)
        
        # 3. Label usage over time
        ax = axes[0, 2]
        # Track top 5 labels over time
        top_5_labels = list(label_counts.most_common(5))
        monthly_label_counts = defaultdict(lambda: defaultdict(int))
        
        for _, row in self.df.iterrows():
            month = row['created_at'].to_period('M')
            for label in row['label_names']:
                if label in [l[0] for l in top_5_labels]:
                    monthly_label_counts[month][label] += 1
        
        months = sorted(monthly_label_counts.keys())
        for label, _ in top_5_labels:
            counts = [monthly_label_counts[m][label] for m in months]
            ax.plot(range(len(months)), counts, marker='o', label=label, linewidth=2)
        
        ax.set_xticks(range(0, len(months), max(1, len(months)//10)))
        ax.set_xticklabels([str(m) for m in months[::max(1, len(months)//10)]], rotation=45, ha='right')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Issues')
        ax.set_title('Top 5 Labels Usage Trend')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # 4. Label effectiveness (closure rate)
        ax = axes[1, 0]
        label_effectiveness = {}
        for label, count in label_counts.most_common(15):
            if count >= 20:  # Minimum sample size
                mask = self.df['label_names'].apply(lambda x: label in x)
                labeled_issues = self.df[mask]
                closure_rate = labeled_issues['is_closed'].mean()
                avg_resolution = labeled_issues['resolution_days'].mean()
                label_effectiveness[label] = {
                    'closure_rate': closure_rate,
                    'count': count,
                    'avg_resolution': avg_resolution
                }
        
        if label_effectiveness:
            eff_df = pd.DataFrame(label_effectiveness).T.sort_values('closure_rate', ascending=False)
            x = np.arange(len(eff_df))
            ax.bar(x, eff_df['closure_rate'], color='lightgreen', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(eff_df.index, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Closure Rate')
            ax.set_title('Label Effectiveness (Closure Rate)')
            ax.set_ylim(0, 1.1)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add count annotations
            for i, (idx, row) in enumerate(eff_df.iterrows()):
                ax.text(i, 0.05, f'n={int(row["count"])}', ha='center', fontsize=7, rotation=90)
        
        # 5. Label co-occurrence heatmap
        ax = axes[1, 1]
        # Calculate co-occurrence matrix for top labels
        top_10_labels = [l[0] for l in label_counts.most_common(10)]
        cooccurrence = np.zeros((len(top_10_labels), len(top_10_labels)))
        
        for labels in self.df['label_names']:
            for i, label1 in enumerate(top_10_labels):
                for j, label2 in enumerate(top_10_labels):
                    if label1 in labels and label2 in labels:
                        cooccurrence[i, j] += 1
        
        # Normalize by diagonal (make it correlation-like)
        for i in range(len(top_10_labels)):
            if cooccurrence[i, i] > 0:
                cooccurrence[i, :] = cooccurrence[i, :] / cooccurrence[i, i]
                cooccurrence[:, i] = cooccurrence[:, i] / cooccurrence[i, i]
        
        sns.heatmap(cooccurrence, xticklabels=top_10_labels, yticklabels=top_10_labels,
                   cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Co-occurrence Rate'})
        ax.set_title('Label Co-occurrence Matrix')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        # 6. Label diversity metrics
        ax = axes[1, 2]
        # Calculate entropy and other diversity metrics
        label_probs = np.array(list(label_counts.values())) / sum(label_counts.values())
        entropy = -sum(p * np.log(p) for p in label_probs if p > 0)
        
        # Unique labels per month
        monthly_unique = defaultdict(set)
        for _, row in self.df.iterrows():
            month = row['created_at'].to_period('M')
            monthly_unique[month].update(row['label_names'])
        
        months = sorted(monthly_unique.keys())
        unique_counts = [len(monthly_unique[m]) for m in months]
        
        ax.plot(range(len(months)), unique_counts, marker='o', linewidth=2)
        ax.set_xticks(range(0, len(months), max(1, len(months)//10)))
        ax.set_xticklabels([str(m) for m in months[::max(1, len(months)//10)]], rotation=45, ha='right')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Unique Labels')
        ax.set_title(f'Label Diversity Over Time (Entropy: {entropy:.2f})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_label_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['distribution'] = {
            'total_unique_labels': len(label_counts),
            'label_entropy': entropy,
            'most_common_label': label_counts.most_common(1)[0] if label_counts else None,
            'avg_labels_per_issue': self.df['label_count'].mean(),
            'no_label_ratio': (self.df['label_count'] == 0).mean()
        }
        
    def analyze_complexity_indicators(self):
        """Analyze various complexity indicators"""
        print(f"[{self.framework}] Analyzing complexity indicators...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.framework} - Issue Complexity Analysis', fontsize=16, y=1.02)
        
        # 1. Comment distribution by complexity
        ax = axes[0, 0]
        # Define complexity buckets
        self.df['complexity_bucket'] = pd.cut(self.df['comments'], 
                                              bins=[-1, 0, 5, 20, 100, 1000],
                                              labels=['No discussion', 'Low (1-5)', 'Medium (6-20)', 
                                                     'High (21-100)', 'Very High (100+)'])
        
        complexity_dist = self.df['complexity_bucket'].value_counts()
        ax.pie(complexity_dist.values, labels=complexity_dist.index, autopct='%1.1f%%', 
               colors=plt.cm.Blues(np.linspace(0.3, 0.9, len(complexity_dist))))
        ax.set_title('Issue Complexity by Discussion Level')
        
        # 2. Complexity vs Resolution Time
        ax = axes[0, 1]
        resolved_df = self.df[self.df['resolution_days'].notna()].copy()
        if len(resolved_df) > 0:
            # Remove extreme outliers for visualization
            resolved_df = resolved_df[resolved_df['resolution_days'] < resolved_df['resolution_days'].quantile(0.95)]
            
            scatter = ax.scatter(resolved_df['comments'], resolved_df['resolution_days'],
                               c=resolved_df['label_count'], cmap='viridis', alpha=0.5)
            ax.set_xlabel('Number of Comments')
            ax.set_ylabel('Resolution Time (days)')
            ax.set_title('Comments vs Resolution Time')
            ax.set_xscale('symlog')
            ax.set_yscale('log')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Number of Labels')
            ax.grid(True, alpha=0.3)
        
        # 3. Multi-dimensional complexity score
        ax = axes[0, 2]
        # Create complexity score
        self.df['complexity_score'] = (
            self.df['comments'].fillna(0) / self.df['comments'].max() * 0.3 +
            self.df['label_count'] / self.df['label_count'].max() * 0.2 +
            self.df['total_reactions'].fillna(0) / self.df['total_reactions'].max() * 0.2 +
            (self.df['body'].str.len().fillna(0) / self.df['body'].str.len().max() * 0.3)
        )
        
        ax.hist(self.df['complexity_score'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(self.df['complexity_score'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {self.df["complexity_score"].mean():.3f}')
        ax.set_xlabel('Complexity Score (0-1)')
        ax.set_ylabel('Number of Issues')
        ax.set_title('Issue Complexity Score Distribution')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # 4. Complexity by state reason
        ax = axes[1, 0]
        state_complexity = self.df.groupby('state_reason').agg({
            'comments': 'mean',
            'label_count': 'mean',
            'complexity_score': 'mean',
            'number': 'count'
        }).sort_values('complexity_score', ascending=False)
        
        if len(state_complexity) > 0:
            x = np.arange(len(state_complexity))
            width = 0.35
            
            ax.bar(x - width/2, state_complexity['comments'], width, label='Avg Comments', alpha=0.7)
            ax.bar(x + width/2, state_complexity['label_count'] * 10, width, 
                   label='Avg Labels (×10)', alpha=0.7)
            
            ax.set_xticks(x)
            ax.set_xticklabels(state_complexity.index, rotation=45, ha='right')
            ax.set_ylabel('Count')
            ax.set_title('Complexity by Resolution Type')
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add sample sizes
            for i, (idx, row) in enumerate(state_complexity.iterrows()):
                ax.text(i, 0.5, f'n={int(row["number"])}', ha='center', fontsize=8)
        
        # 5. Reaction patterns by complexity
        ax = axes[1, 1]
        # Group by complexity bucket and calculate reaction stats
        reaction_by_complexity = self.df.groupby('complexity_bucket').agg({
            'total_reactions': ['mean', 'sum', 'count'],
            'is_closed': 'mean'
        })
        
        if len(reaction_by_complexity) > 0:
            x = np.arange(len(reaction_by_complexity))
            ax.bar(x, reaction_by_complexity['total_reactions']['mean'], alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(reaction_by_complexity.index, rotation=45, ha='right')
            ax.set_ylabel('Average Total Reactions')
            ax.set_title('User Reactions by Issue Complexity')
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add closure rate as line
            ax2 = ax.twinx()
            ax2.plot(x, reaction_by_complexity['is_closed']['mean'], 'ro-', label='Closure Rate')
            ax2.set_ylabel('Closure Rate')
            ax2.set_ylim(0, 1.1)
        
        # 6. Complexity trends over time
        ax = axes[1, 2]
        monthly_complexity = self.df.groupby(pd.Grouper(key='created_at', freq='M')).agg({
            'complexity_score': 'mean',
            'comments': 'mean',
            'label_count': 'mean'
        })
        
        if len(monthly_complexity) > 5:
            ax.plot(monthly_complexity.index, monthly_complexity['complexity_score'], 
                   marker='o', label='Complexity Score', linewidth=2)
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Complexity Score')
            ax.set_title('Issue Complexity Trend Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add rolling average
            rolling_avg = monthly_complexity['complexity_score'].rolling(window=3, center=True).mean()
            ax.plot(monthly_complexity.index, rolling_avg, '--', alpha=0.7, 
                   label='3-month Rolling Avg')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['complexity'] = {
            'avg_comments': self.df['comments'].mean(),
            'median_comments': self.df['comments'].median(),
            'high_complexity_ratio': (self.df['complexity_score'] > 0.7).mean(),
            'no_discussion_ratio': (self.df['comments'] == 0).mean(),
            'avg_complexity_score': self.df['complexity_score'].mean()
        }
        
    def analyze_label_effectiveness(self):
        """Deep dive into label effectiveness and patterns"""
        print(f"[{self.framework}] Analyzing label effectiveness...")
        
        # Prepare data for analysis
        all_labels = []
        for labels in self.df['label_names']:
            all_labels.extend(labels)
        label_counts = Counter(all_labels)
        
        # Only analyze labels with sufficient data
        significant_labels = [label for label, count in label_counts.items() if count >= 30]
        
        if len(significant_labels) >= 5:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{self.framework} - Label Effectiveness Analysis', fontsize=16, y=1.02)
            
            # 1. Label resolution time comparison
            ax = axes[0, 0]
            label_resolution_times = {}
            
            for label in significant_labels[:15]:  # Top 15 labels
                mask = self.df['label_names'].apply(lambda x: label in x)
                labeled_resolved = self.df[mask & self.df['resolution_days'].notna()]
                if len(labeled_resolved) > 10:
                    label_resolution_times[label] = {
                        'median': labeled_resolved['resolution_days'].median(),
                        'mean': labeled_resolved['resolution_days'].mean(),
                        'count': len(labeled_resolved)
                    }
            
            if label_resolution_times:
                res_df = pd.DataFrame(label_resolution_times).T.sort_values('median')
                x = np.arange(len(res_df))
                ax.bar(x, res_df['median'], alpha=0.7, label='Median')
                ax.errorbar(x, res_df['median'], 
                           yerr=(res_df['mean'] - res_df['median']).abs(), 
                           fmt='none', color='black', alpha=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(res_df.index, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Resolution Time (days)')
                ax.set_title('Resolution Time by Label')
                ax.grid(True, axis='y', alpha=0.3)
            
            # 2. Label combination analysis
            ax = axes[0, 1]
            # Find most common label combinations
            label_combos = Counter()
            for labels in self.df['label_names']:
                if len(labels) >= 2:
                    for combo in combinations(sorted(labels), 2):
                        if all(label in significant_labels for label in combo):
                            label_combos[combo] += 1
            
            if label_combos:
                top_combos = label_combos.most_common(10)
                combo_labels = [f"{c[0][0]}\n+\n{c[0][1]}" for c in top_combos]
                combo_counts = [c[1] for c in top_combos]
                
                y_pos = np.arange(len(top_combos))
                ax.barh(y_pos, combo_counts, color='lightcoral', alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(combo_labels, fontsize=8)
                ax.set_xlabel('Number of Issues')
                ax.set_title('Top Label Combinations')
                ax.grid(True, axis='x', alpha=0.3)
            
            # 3. Label lifecycle analysis
            ax = axes[1, 0]
            # Track when labels are typically added/removed
            label_lifecycle = defaultdict(lambda: {'created': [], 'closed': []})
            
            for _, row in self.df.iterrows():
                for label in row['label_names']:
                    if label in significant_labels[:10]:  # Top 10 labels
                        label_lifecycle[label]['created'].append(row['created_at'])
                        if row['is_closed'] and pd.notna(row['closed_at']):
                            label_lifecycle[label]['closed'].append(row['closed_at'])
            
            # Calculate average age when closed
            label_ages = {}
            for label, dates in label_lifecycle.items():
                if dates['closed']:
                    ages = [(c - cr).days for cr, c in zip(dates['created'][:len(dates['closed'])], 
                                                           dates['closed'])]
                    label_ages[label] = np.mean(ages)
            
            if label_ages:
                sorted_ages = sorted(label_ages.items(), key=lambda x: x[1])
                labels = [x[0] for x in sorted_ages]
                ages = [x[1] for x in sorted_ages]
                
                ax.bar(range(len(labels)), ages, color='lightgreen', alpha=0.7)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Average Age at Closure (days)')
                ax.set_title('Label Resolution Speed')
                ax.grid(True, axis='y', alpha=0.3)
            
            # 4. Label predictive power
            ax = axes[1, 1]
            # Calculate correlation between labels and outcomes
            label_correlations = {}
            
            for label in significant_labels[:20]:
                mask = self.df['label_names'].apply(lambda x: label in x)
                if mask.sum() > 30:
                    # Calculate various correlations
                    labeled_df = self.df[mask]
                    unlabeled_df = self.df[~mask]
                    
                    closure_diff = labeled_df['is_closed'].mean() - unlabeled_df['is_closed'].mean()
                    comment_diff = labeled_df['comments'].mean() - unlabeled_df['comments'].mean()
                    
                    label_correlations[label] = {
                        'closure_impact': closure_diff,
                        'comment_impact': comment_diff / self.df['comments'].mean(),  # Normalized
                        'frequency': mask.sum()
                    }
            
            if label_correlations:
                corr_df = pd.DataFrame(label_correlations).T
                
                # Create scatter plot
                scatter = ax.scatter(corr_df['closure_impact'], 
                                   corr_df['comment_impact'],
                                   s=corr_df['frequency'] / 5,  # Size by frequency
                                   alpha=0.6,
                                   c=corr_df['frequency'],
                                   cmap='viridis')
                
                # Add labels for interesting points
                for idx, row in corr_df.iterrows():
                    if abs(row['closure_impact']) > 0.1 or abs(row['comment_impact']) > 0.5:
                        ax.annotate(idx, (row['closure_impact'], row['comment_impact']), 
                                   fontsize=7, alpha=0.7)
                
                ax.axhline(0, color='black', linestyle='-', alpha=0.3)
                ax.axvline(0, color='black', linestyle='-', alpha=0.3)
                ax.set_xlabel('Impact on Closure Rate')
                ax.set_ylabel('Impact on Comment Activity (normalized)')
                ax.set_title('Label Impact on Issue Outcomes')
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Frequency')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{self.framework}_label_effectiveness.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Store effectiveness results
            self.results['effectiveness'] = {
                'labels_analyzed': len(significant_labels),
                'avg_resolution_time_variation': np.std([v['median'] for v in label_resolution_times.values()]) 
                                               if label_resolution_times else 0,
                'most_effective_label': min(label_resolution_times.items(), key=lambda x: x[1]['median'])[0] 
                                       if label_resolution_times else None,
                'common_combinations': len(label_combos)
            }
    
    def analyze_label_evolution(self):
        """Analyze how label usage evolves over time"""
        print(f"[{self.framework}] Analyzing label evolution...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.framework} - Label Evolution Analysis', fontsize=16, y=1.02)
        
        # 1. New label introduction over time
        ax = axes[0, 0]
        seen_labels = set()
        new_labels_timeline = []
        
        for _, row in self.df.sort_values('created_at').iterrows():
            month = row['created_at'].to_period('M')
            new_in_issue = set(row['label_names']) - seen_labels
            if new_in_issue:
                new_labels_timeline.append((month, len(new_in_issue)))
                seen_labels.update(new_in_issue)
        
        if new_labels_timeline:
            monthly_new = defaultdict(int)
            for month, count in new_labels_timeline:
                monthly_new[month] += count
            
            months = sorted(monthly_new.keys())
            counts = [monthly_new[m] for m in months]
            
            ax.bar(range(len(months)), counts, color='lightblue', alpha=0.7)
            ax.set_xticks(range(0, len(months), max(1, len(months)//10)))
            ax.set_xticklabels([str(m) for m in months[::max(1, len(months)//10)]], rotation=45, ha='right')
            ax.set_xlabel('Month')
            ax.set_ylabel('New Labels Introduced')
            ax.set_title('New Label Introduction Rate')
            ax.grid(True, axis='y', alpha=0.3)
        
        # 2. Label popularity lifecycle
        ax = axes[0, 1]
        # Track label usage percentage over time
        all_labels = []
        for labels in self.df['label_names']:
            all_labels.extend(labels)
        label_counts = Counter(all_labels)
        
        # Select labels that were popular at some point
        interesting_labels = []
        monthly_label_pct = defaultdict(lambda: defaultdict(float))
        
        for _, row in self.df.iterrows():
            month = row['created_at'].to_period('M')
            for label in row['label_names']:
                monthly_label_pct[month][label] += 1
        
        # Normalize to percentages
        for month in monthly_label_pct:
            total = sum(monthly_label_pct[month].values())
            if total > 0:
                for label in monthly_label_pct[month]:
                    monthly_label_pct[month][label] /= total
        
        # Find labels with interesting lifecycles (peaked and declined)
        for label in label_counts:
            if label_counts[label] >= 50:  # Minimum threshold
                monthly_pcts = [monthly_label_pct[m].get(label, 0) for m in sorted(monthly_label_pct.keys())]
                if max(monthly_pcts) > 0.05:  # Was significant at some point
                    peak_idx = monthly_pcts.index(max(monthly_pcts))
                    if peak_idx < len(monthly_pcts) - 6:  # Peaked before the last 6 months
                        if monthly_pcts[-1] < max(monthly_pcts) * 0.5:  # Declined significantly
                            interesting_labels.append(label)
        
        # Plot lifecycle of interesting labels
        months = sorted(monthly_label_pct.keys())
        for label in interesting_labels[:5]:  # Top 5 most interesting
            pcts = [monthly_label_pct[m].get(label, 0) * 100 for m in months]
            ax.plot(range(len(months)), pcts, marker='o', label=label, linewidth=2)
        
        if interesting_labels:
            ax.set_xticks(range(0, len(months), max(1, len(months)//10)))
            ax.set_xticklabels([str(m) for m in months[::max(1, len(months)//10)]], rotation=45, ha='right')
            ax.set_xlabel('Month')
            ax.set_ylabel('% of Issues with Label')
            ax.set_title('Label Popularity Lifecycle')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
        
        # 3. Label stability analysis
        ax = axes[1, 0]
        # Calculate label usage variance over time
        label_stability = {}
        
        for label in label_counts:
            if label_counts[label] >= 30:
                monthly_counts = []
                for month in sorted(monthly_label_pct.keys()):
                    monthly_counts.append(monthly_label_pct[month].get(label, 0))
                
                if len(monthly_counts) > 6:
                    # Calculate coefficient of variation
                    mean_usage = np.mean(monthly_counts)
                    if mean_usage > 0:
                        cv = np.std(monthly_counts) / mean_usage
                        label_stability[label] = {
                            'cv': cv,
                            'count': label_counts[label],
                            'mean_usage': mean_usage
                        }
        
        if label_stability:
            # Sort by stability (lower CV = more stable)
            stable_labels = sorted(label_stability.items(), key=lambda x: x[1]['cv'])[:10]
            unstable_labels = sorted(label_stability.items(), key=lambda x: x[1]['cv'], reverse=True)[:10]
            
            labels = [l[0] for l in stable_labels] + [l[0] for l in unstable_labels]
            cvs = [l[1]['cv'] for l in stable_labels] + [l[1]['cv'] for l in unstable_labels]
            colors = ['green'] * len(stable_labels) + ['red'] * len(unstable_labels)
            
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, cvs, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Coefficient of Variation')
            ax.set_title('Label Usage Stability (Green=Stable, Red=Unstable)')
            ax.grid(True, axis='x', alpha=0.3)
        
        # 4. Emerging vs declining labels
        ax = axes[1, 1]
        # Compare first half vs second half of data
        mid_date = self.df['created_at'].min() + (self.df['created_at'].max() - self.df['created_at'].min()) / 2
        first_half = self.df[self.df['created_at'] < mid_date]
        second_half = self.df[self.df['created_at'] >= mid_date]
        
        first_labels = Counter()
        second_labels = Counter()
        
        for labels in first_half['label_names']:
            first_labels.update(labels)
        for labels in second_half['label_names']:
            second_labels.update(labels)
        
        # Normalize by number of issues
        for label in first_labels:
            first_labels[label] = first_labels[label] / len(first_half)
        for label in second_labels:
            second_labels[label] = second_labels[label] / len(second_half)
        
        # Find emerging and declining labels
        emerging = []
        declining = []
        
        for label in set(first_labels.keys()) | set(second_labels.keys()):
            first_rate = first_labels.get(label, 0)
            second_rate = second_labels.get(label, 0)
            
            if second_rate > first_rate * 1.5 and second_rate > 0.01:  # 50% increase and meaningful
                emerging.append((label, second_rate / (first_rate + 0.001)))
            elif first_rate > second_rate * 1.5 and first_rate > 0.01:  # 50% decrease
                declining.append((label, first_rate / (second_rate + 0.001)))
        
        # Plot top emerging and declining
        emerging.sort(key=lambda x: x[1], reverse=True)
        declining.sort(key=lambda x: x[1], reverse=True)
        
        all_labels = [(l, r, 'emerging') for l, r in emerging[:5]] + \
                    [(l, r, 'declining') for l, r in declining[:5]]
        
        if all_labels:
            labels = [l[0] for l in all_labels]
            ratios = [l[1] for l in all_labels]
            colors = ['green' if l[2] == 'emerging' else 'red' for l in all_labels]
            
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, ratios, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Growth/Decline Ratio')
            ax.set_title('Emerging (Green) vs Declining (Red) Labels')
            ax.axvline(1, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_label_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store evolution results
        self.results['evolution'] = {
            'total_labels_seen': len(seen_labels),
            'labels_still_active': len([l for l in second_labels if second_labels[l] > 0.01]),
            'emerging_labels': len(emerging),
            'declining_labels': len(declining),
            'label_introduction_rate': len(seen_labels) / len(months) if months else 0
        }
    
    def generate_report(self):
        """Generate label and complexity analysis report"""
        report = f"""# {self.framework} - Label and Complexity Analysis Report

## Label Distribution
- Total unique labels: {self.results['distribution']['total_unique_labels']}
- Label entropy: {self.results['distribution']['label_entropy']:.3f}
- Average labels per issue: {self.results['distribution']['avg_labels_per_issue']:.2f}
- Issues without labels: {self.results['distribution']['no_label_ratio']*100:.1f}%
"""
        
        if self.results['distribution']['most_common_label']:
            report += f"- Most common label: {self.results['distribution']['most_common_label'][0]} ({self.results['distribution']['most_common_label'][1]} issues)\n"
        
        report += f"""
## Issue Complexity
- Average comments per issue: {self.results['complexity']['avg_comments']:.2f}
- Median comments: {self.results['complexity']['median_comments']:.0f}
- High complexity issues: {self.results['complexity']['high_complexity_ratio']*100:.1f}%
- No discussion issues: {self.results['complexity']['no_discussion_ratio']*100:.1f}%
- Average complexity score: {self.results['complexity']['avg_complexity_score']:.3f}
"""
        
        if 'effectiveness' in self.results:
            report += f"""
## Label Effectiveness
- Labels analyzed: {self.results['effectiveness']['labels_analyzed']}
- Resolution time variation: {self.results['effectiveness']['avg_resolution_time_variation']:.1f} days
- Most effective label: {self.results['effectiveness']['most_effective_label']}
- Common label combinations: {self.results['effectiveness']['common_combinations']}
"""
        
        if 'evolution' in self.results:
            report += f"""
## Label Evolution
- Total labels introduced: {self.results['evolution']['total_labels_seen']}
- Currently active labels: {self.results['evolution']['labels_still_active']}
- Emerging labels: {self.results['evolution']['emerging_labels']}
- Declining labels: {self.results['evolution']['declining_labels']}
- Label introduction rate: {self.results['evolution']['label_introduction_rate']:.2f} per month
"""
        
        return report
    
    def run(self):
        """Run all label and complexity analyses"""
        self.load_data()
        self.analyze_label_distribution()
        self.analyze_complexity_indicators()
        self.analyze_label_effectiveness()
        self.analyze_label_evolution()
        
        # Save report
        report = self.generate_report()
        with open(self.output_dir / f'{self.framework}_label_complexity_report.md', 'w') as f:
            f.write(report)
        
        # Save data
        with open(self.output_dir / f'{self.framework}_label_complexity_data.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return self.results


if __name__ == "__main__":
    # Example usage
    analyzer = LabelComplexityAnalyzer(
        'vllm',
        '/root/yunwei37/vllm-exp/bug-study/data/vllm_all_issues.json',
        '/root/yunwei37/vllm-exp/bug-study/analysis/results/label_complexity'
    )
    analyzer.run()