#!/usr/bin/env python3
"""
Cross-Framework Comparison Module: Compare patterns across vLLM, SGLang, and llama.cpp
Research Questions:
- Framework maturity comparison
- Community health metrics
- Issue pattern differences
- Best practices identification
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class CrossFrameworkAnalyzer:
    def __init__(self, data_paths, output_dir):
        self.data_paths = data_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frameworks = {}
        self.results = {}
        
    def load_all_data(self):
        """Load data for all frameworks"""
        for name, path in self.data_paths.items():
            print(f"Loading {name} data...")
            with open(path, 'r') as f:
                data = json.load(f)
            
            items = data.get('items', data if isinstance(data, list) else [])
            df = pd.DataFrame(items)
            
            # Basic preprocessing
            for col in ['created_at', 'updated_at', 'closed_at']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            df['is_closed'] = df['state'] == 'closed'
            df['label_count'] = df['labels'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            df['user_login'] = df['user'].apply(lambda x: x.get('login') if isinstance(x, dict) else None)
            
            # Calculate resolution time
            mask = df['is_closed'] & df['closed_at'].notna()
            df.loc[mask, 'resolution_days'] = (
                df.loc[mask, 'closed_at'] - df.loc[mask, 'created_at']
            ).dt.total_seconds() / 86400
            
            self.frameworks[name] = df
            print(f"Loaded {len(df)} issues for {name}")
    
    def compare_basic_metrics(self):
        """Compare basic metrics across frameworks"""
        print("Comparing basic metrics across frameworks...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cross-Framework Comparison - Basic Metrics', fontsize=16, y=1.02)
        
        # Collect metrics
        metrics = {}
        for name, df in self.frameworks.items():
            metrics[name] = {
                'total_issues': len(df),
                'open_issues': (~df['is_closed']).sum(),
                'closed_issues': df['is_closed'].sum(),
                'closure_rate': df['is_closed'].mean(),
                'unique_users': df['user_login'].nunique(),
                'avg_comments': df['comments'].mean(),
                'avg_labels': df['label_count'].mean(),
                'median_resolution_days': df['resolution_days'].median()
            }
        
        metrics_df = pd.DataFrame(metrics).T
        
        # 1. Issue volume comparison
        ax = axes[0, 0]
        x = np.arange(len(metrics_df))
        width = 0.35
        
        ax.bar(x - width/2, metrics_df['open_issues'], width, label='Open', color='#e74c3c', alpha=0.7)
        ax.bar(x + width/2, metrics_df['closed_issues'], width, label='Closed', color='#2ecc71', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df.index)
        ax.set_ylabel('Number of Issues')
        ax.set_title('Issue Volume by Framework')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add total count annotations
        for i, (idx, row) in enumerate(metrics_df.iterrows()):
            ax.text(i, row['total_issues'] + 100, f"{int(row['total_issues'])}", 
                   ha='center', fontsize=10, fontweight='bold')
        
        # 2. Closure rate comparison
        ax = axes[0, 1]
        colors = ['#3498db', '#e74c3c', '#f39c12']
        bars = ax.bar(metrics_df.index, metrics_df['closure_rate'], color=colors, alpha=0.7)
        ax.set_ylabel('Closure Rate')
        ax.set_title('Issue Closure Rate')
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, metrics_df['closure_rate']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{rate:.1%}', ha='center', va='bottom')
        
        # 3. Community size
        ax = axes[0, 2]
        ax.bar(metrics_df.index, metrics_df['unique_users'], color='lightcoral', alpha=0.7)
        ax.set_ylabel('Number of Unique Users')
        ax.set_title('Community Size (Unique Contributors)')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 4. Resolution time comparison
        ax = axes[1, 0]
        resolution_data = []
        labels = []
        for name, df in self.frameworks.items():
            res_days = df['resolution_days'].dropna()
            if len(res_days) > 0:
                resolution_data.append(res_days)
                labels.append(name)
        
        if resolution_data:
            bp = ax.boxplot(resolution_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(labels)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Days to Resolution')
            ax.set_title('Resolution Time Distribution')
            ax.set_yscale('log')
            ax.grid(True, axis='y', alpha=0.3)
        
        # 5. Activity metrics
        ax = axes[1, 1]
        activity_metrics = ['avg_comments', 'avg_labels']
        x = np.arange(len(metrics_df))
        width = 0.35
        
        for i, metric in enumerate(activity_metrics):
            offset = width * (i - 0.5)
            ax.bar(x + offset, metrics_df[metric], width, 
                  label=metric.replace('avg_', 'Avg ').replace('_', ' ').title(), alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df.index)
        ax.set_ylabel('Average Count')
        ax.set_title('Issue Activity Metrics')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # 6. Project age and maturity
        ax = axes[1, 2]
        age_metrics = {}
        for name, df in self.frameworks.items():
            age_days = (df['created_at'].max() - df['created_at'].min()).days
            age_metrics[name] = {
                'age_days': age_days,
                'issues_per_day': len(df) / age_days,
                'active_days': df['created_at'].dt.date.nunique()
            }
        
        age_df = pd.DataFrame(age_metrics).T
        
        # Create scatter plot of age vs issues per day
        scatter = ax.scatter(age_df['age_days'], age_df['issues_per_day'], 
                           s=metrics_df['unique_users']*2, alpha=0.6, c=range(len(age_df)))
        
        for i, (idx, row) in enumerate(age_df.iterrows()):
            ax.annotate(idx, (row['age_days'], row['issues_per_day']), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Project Age (days)')
        ax.set_ylabel('Issues per Day')
        ax.set_title('Project Maturity (bubble size = community size)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_framework_basic_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['basic_metrics'] = metrics
        
    def compare_temporal_patterns(self):
        """Compare temporal patterns across frameworks"""
        print("Comparing temporal patterns...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cross-Framework Temporal Pattern Comparison', fontsize=16, y=1.02)
        
        # 1. Issue creation trends
        ax = axes[0, 0]
        for name, df in self.frameworks.items():
            monthly_counts = df.groupby(pd.Grouper(key='created_at', freq='M')).size()
            # Normalize by max to compare trends
            normalized_counts = monthly_counts / monthly_counts.max()
            ax.plot(normalized_counts.index, normalized_counts.values, 
                   marker='o', label=name, linewidth=2, markersize=4)
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Normalized Issue Count')
        ax.set_title('Issue Creation Trends (Normalized)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Day of week patterns
        ax = axes[0, 1]
        dow_patterns = {}
        for name, df in self.frameworks.items():
            dow_dist = df['created_at'].dt.dayofweek.value_counts(normalize=True).sort_index()
            dow_patterns[name] = dow_dist
        
        dow_df = pd.DataFrame(dow_patterns)
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        x = np.arange(7)
        width = 0.25
        for i, (name, values) in enumerate(dow_df.items()):
            offset = width * (i - 1)
            ax.bar(x + offset, values, width, label=name, alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(dow_names)
        ax.set_ylabel('Proportion of Issues')
        ax.set_title('Day of Week Distribution')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # 3. Hour of day patterns
        ax = axes[1, 0]
        for name, df in self.frameworks.items():
            hourly_dist = df['created_at'].dt.hour.value_counts(normalize=True).sort_index()
            ax.plot(hourly_dist.index, hourly_dist.values * 100, 
                   marker='o', label=name, linewidth=2, markersize=4)
        
        ax.set_xlabel('Hour of Day (UTC)')
        ax.set_ylabel('% of Issues')
        ax.set_title('Hour of Day Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 3))
        
        # 4. Resolution time trends
        ax = axes[1, 1]
        for name, df in self.frameworks.items():
            monthly_resolution = df[df['resolution_days'].notna()].groupby(
                pd.Grouper(key='created_at', freq='M')
            )['resolution_days'].median()
            
            if len(monthly_resolution) > 5:  # Need sufficient data
                # Smooth with rolling average
                smoothed = monthly_resolution.rolling(window=3, center=True).mean()
                ax.plot(smoothed.index, smoothed.values, 
                       label=name, linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Median Resolution Days')
        ax.set_title('Resolution Time Trends (3-month rolling average)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_framework_temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def compare_community_health(self):
        """Compare community health metrics across frameworks"""
        print("Comparing community health metrics...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cross-Framework Community Health Comparison', fontsize=16, y=1.02)
        
        # 1. User contribution distribution (Lorenz curves)
        ax = axes[0, 0]
        for name, df in self.frameworks.items():
            user_counts = df['user_login'].value_counts()
            sorted_contributions = np.sort(user_counts.values)
            cumsum = np.cumsum(sorted_contributions)
            cumsum_normalized = cumsum / cumsum[-1]
            user_percentiles = np.arange(1, len(sorted_contributions) + 1) / len(sorted_contributions)
            
            ax.plot(user_percentiles, cumsum_normalized, linewidth=2, label=name)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality')
        ax.set_xlabel('Cumulative % of Users')
        ax.set_ylabel('Cumulative % of Contributions')
        ax.set_title('User Contribution Inequality (Lorenz Curves)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. New vs returning users
        ax = axes[0, 1]
        retention_metrics = {}
        
        for name, df in self.frameworks.items():
            # Calculate monthly new/returning users
            seen_users = set()
            monthly_users = defaultdict(lambda: {'new': 0, 'returning': 0})
            
            for _, row in df.sort_values('created_at').iterrows():
                month = row['created_at'].to_period('M')
                user = row['user_login']
                if user in seen_users:
                    monthly_users[month]['returning'] += 1
                else:
                    monthly_users[month]['new'] += 1
                    seen_users.add(user)
            
            # Calculate average retention rate
            total_new = sum(m['new'] for m in monthly_users.values())
            total_returning = sum(m['returning'] for m in monthly_users.values())
            retention_rate = total_returning / (total_new + total_returning) if total_new + total_returning > 0 else 0
            
            retention_metrics[name] = {
                'retention_rate': retention_rate,
                'total_users': len(seen_users),
                'avg_contributions': len(df) / len(seen_users) if len(seen_users) > 0 else 0
            }
        
        ret_df = pd.DataFrame(retention_metrics).T
        
        x = np.arange(len(ret_df))
        ax.bar(x, ret_df['retention_rate'], color='lightgreen', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(ret_df.index)
        ax.set_ylabel('User Retention Rate')
        ax.set_title('Community Retention (Returning User Rate)')
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for i, rate in enumerate(ret_df['retention_rate']):
            ax.text(i, rate + 0.02, f'{rate:.1%}', ha='center', va='bottom')
        
        # 3. Response time distribution
        ax = axes[0, 2]
        response_data = {}
        
        for name, df in self.frameworks.items():
            response_hours = (df['updated_at'] - df['created_at']).dt.total_seconds() / 3600
            response_hours = response_hours[(response_hours > 0) & (response_hours < 24*7)]  # Within a week
            
            if len(response_hours) > 0:
                response_data[name] = {
                    'median': response_hours.median(),
                    'p90': response_hours.quantile(0.9),
                    'under_24h': (response_hours < 24).mean()
                }
        
        if response_data:
            resp_df = pd.DataFrame(response_data).T
            
            x = np.arange(len(resp_df))
            width = 0.35
            
            ax.bar(x - width/2, resp_df['median'], width, label='Median', alpha=0.7)
            ax.bar(x + width/2, resp_df['p90'], width, label='90th Percentile', alpha=0.7)
            
            ax.set_xticks(x)
            ax.set_xticklabels(resp_df.index)
            ax.set_ylabel('Hours')
            ax.set_title('Response Time Metrics')
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
        
        # 4. Author association distribution
        ax = axes[1, 0]
        association_data = {}
        
        for name, df in self.frameworks.items():
            assoc_dist = df['author_association'].value_counts(normalize=True)
            association_data[name] = assoc_dist
        
        # Get all unique associations
        all_associations = set()
        for dist in association_data.values():
            all_associations.update(dist.index)
        
        # Create stacked bar chart
        bottom = np.zeros(len(association_data))
        for assoc in sorted(all_associations):
            values = [association_data[fw].get(assoc, 0) for fw in association_data.keys()]
            ax.bar(range(len(association_data)), values, bottom=bottom, 
                  label=assoc, alpha=0.7)
            bottom += values
        
        ax.set_xticks(range(len(association_data)))
        ax.set_xticklabels(list(association_data.keys()))
        ax.set_ylabel('Proportion')
        ax.set_title('Author Association Distribution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 5. Issue quality metrics
        ax = axes[1, 1]
        quality_metrics = {}
        
        for name, df in self.frameworks.items():
            quality_metrics[name] = {
                'has_labels': (df['label_count'] > 0).mean(),
                'has_comments': (df['comments'] > 0).mean(),
                'avg_body_length': df['body'].str.len().mean() if 'body' in df.columns else 0,
                'quick_resolution': (df['resolution_days'] < 7).sum() / df['is_closed'].sum() 
                                   if df['is_closed'].sum() > 0 else 0
            }
        
        qual_df = pd.DataFrame(quality_metrics).T
        
        # Normalize metrics for radar chart effect in bar chart
        metrics_to_show = ['has_labels', 'has_comments', 'quick_resolution']
        x = np.arange(len(qual_df))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_show):
            offset = width * (i - 1)
            ax.bar(x + offset, qual_df[metric], width, 
                  label=metric.replace('_', ' ').title(), alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(qual_df.index)
        ax.set_ylabel('Proportion')
        ax.set_title('Issue Quality Indicators')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # 6. Community growth rate
        ax = axes[1, 2]
        growth_data = {}
        
        for name, df in self.frameworks.items():
            # Calculate monthly unique users
            monthly_users = df.groupby(pd.Grouper(key='created_at', freq='M'))['user_login'].nunique()
            
            if len(monthly_users) > 12:  # Need sufficient history
                # Calculate year-over-year growth
                recent_avg = monthly_users[-6:].mean()  # Last 6 months
                older_avg = monthly_users[-18:-12].mean()  # 12-18 months ago
                growth_rate = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                
                growth_data[name] = {
                    'growth_rate': growth_rate,
                    'recent_users': recent_avg,
                    'trend': 'growing' if growth_rate > 0.1 else 'stable' if growth_rate > -0.1 else 'declining'
                }
        
        if growth_data:
            colors = {'growing': 'green', 'stable': 'blue', 'declining': 'red'}
            
            for i, (name, data) in enumerate(growth_data.items()):
                color = colors[data['trend']]
                ax.bar(i, data['growth_rate'], color=color, alpha=0.7)
                ax.text(i, data['growth_rate'] + 0.02, f"{data['growth_rate']:.1%}", 
                       ha='center', va='bottom')
            
            ax.set_xticks(range(len(growth_data)))
            ax.set_xticklabels(list(growth_data.keys()))
            ax.set_ylabel('Growth Rate')
            ax.set_title('Community Growth Rate (YoY)')
            ax.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_framework_community_health.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['community_health'] = {
            'retention_metrics': retention_metrics,
            'quality_metrics': quality_metrics,
            'growth_data': growth_data
        }
        
    def compare_issue_characteristics(self):
        """Compare issue characteristics across frameworks"""
        print("Comparing issue characteristics...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cross-Framework Issue Characteristics', fontsize=16, y=1.02)
        
        # 1. Label usage patterns
        ax = axes[0, 0]
        label_stats = {}
        
        for name, df in self.frameworks.items():
            all_labels = []
            for labels in df['labels']:
                if isinstance(labels, list):
                    all_labels.extend([l['name'] for l in labels if isinstance(l, dict) and 'name' in l])
            
            label_stats[name] = {
                'unique_labels': len(set(all_labels)),
                'avg_labels_per_issue': df['label_count'].mean(),
                'no_label_rate': (df['label_count'] == 0).mean(),
                'total_label_uses': len(all_labels)
            }
        
        stats_df = pd.DataFrame(label_stats).T
        
        # Normalize metrics for comparison
        metrics = ['unique_labels', 'avg_labels_per_issue']
        x = np.arange(len(stats_df))
        width = 0.35
        
        # Normalize by max value for each metric
        for i, metric in enumerate(metrics):
            values = stats_df[metric] / stats_df[metric].max()
            offset = width * (i - 0.5)
            ax.bar(x + offset, values, width, 
                  label=metric.replace('_', ' ').title(), alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df.index)
        ax.set_ylabel('Normalized Value')
        ax.set_title('Label Usage Patterns (Normalized)')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # 2. Complexity distribution
        ax = axes[0, 1]
        complexity_data = []
        labels = []
        
        for name, df in self.frameworks.items():
            # Create complexity score
            complexity = (
                df['comments'].fillna(0) / df['comments'].max() * 0.5 +
                df['label_count'] / df['label_count'].max() * 0.3 +
                (df['body'].str.len().fillna(0) / df['body'].str.len().max() * 0.2)
                if 'body' in df.columns else 0
            )
            complexity_data.append(complexity)
            labels.append(name)
        
        # Create violin plot
        parts = ax.violinplot(complexity_data, positions=range(len(labels)), showmeans=True)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Complexity Score')
        ax.set_title('Issue Complexity Distribution')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 3. State reason comparison
        ax = axes[1, 0]
        state_reason_data = {}
        
        for name, df in self.frameworks.items():
            closed_df = df[df['is_closed']]
            if len(closed_df) > 0:
                reason_dist = closed_df['state_reason'].value_counts(normalize=True).head(5)
                state_reason_data[name] = reason_dist
        
        # Plot grouped bar chart
        all_reasons = set()
        for dist in state_reason_data.values():
            all_reasons.update(dist.index)
        
        reason_list = sorted(all_reasons)
        x = np.arange(len(reason_list))
        width = 0.25
        
        for i, (name, dist) in enumerate(state_reason_data.items()):
            values = [dist.get(reason, 0) for reason in reason_list]
            offset = width * (i - 1)
            ax.bar(x + offset, values, width, label=name, alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(reason_list, rotation=45, ha='right')
        ax.set_ylabel('Proportion')
        ax.set_title('Top Close Reasons Distribution')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # 4. Resolution efficiency
        ax = axes[1, 1]
        efficiency_metrics = {}
        
        for name, df in self.frameworks.items():
            closed_df = df[df['is_closed']]
            if len(closed_df) > 0:
                efficiency_metrics[name] = {
                    'quick_close_rate': (closed_df['resolution_days'] < 1).sum() / len(closed_df),
                    'moderate_close_rate': ((closed_df['resolution_days'] >= 1) & 
                                          (closed_df['resolution_days'] < 7)).sum() / len(closed_df),
                    'slow_close_rate': (closed_df['resolution_days'] >= 30).sum() / len(closed_df)
                }
        
        if efficiency_metrics:
            eff_df = pd.DataFrame(efficiency_metrics).T
            
            # Create stacked bar chart
            categories = ['quick_close_rate', 'moderate_close_rate', 'slow_close_rate']
            colors = ['green', 'yellow', 'red']
            
            bottom = np.zeros(len(eff_df))
            for cat, color in zip(categories, colors):
                ax.bar(range(len(eff_df)), eff_df[cat], bottom=bottom, 
                      label=cat.replace('_', ' ').title(), color=color, alpha=0.7)
                bottom += eff_df[cat]
            
            ax.set_xticks(range(len(eff_df)))
            ax.set_xticklabels(eff_df.index)
            ax.set_ylabel('Proportion')
            ax.set_title('Resolution Speed Distribution')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_framework_issue_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['issue_characteristics'] = {
            'label_stats': label_stats,
            'state_reason_data': {k: v.to_dict() if hasattr(v, 'to_dict') else v 
                                 for k, v in state_reason_data.items()},
            'efficiency_metrics': efficiency_metrics
        }
        
    def generate_comparison_matrix(self):
        """Generate a comprehensive comparison matrix"""
        print("Generating comparison matrix...")
        
        # Collect all metrics for comparison
        comparison_data = {}
        
        for name in self.frameworks.keys():
            metrics = self.results['basic_metrics'][name]
            
            # Add community health metrics
            if name in self.results['community_health']['retention_metrics']:
                metrics.update({
                    'retention_rate': self.results['community_health']['retention_metrics'][name]['retention_rate'],
                    'avg_contributions_per_user': self.results['community_health']['retention_metrics'][name]['avg_contributions']
                })
            
            # Add quality metrics
            if name in self.results['community_health']['quality_metrics']:
                metrics.update({
                    'has_labels_rate': self.results['community_health']['quality_metrics'][name]['has_labels'],
                    'has_comments_rate': self.results['community_health']['quality_metrics'][name]['has_comments']
                })
            
            comparison_data[name] = metrics
        
        # Create comparison DataFrame
        comp_df = pd.DataFrame(comparison_data).T
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalize data for heatmap (0-1 scale for each metric)
        normalized_df = comp_df.copy()
        for col in comp_df.columns:
            if comp_df[col].std() > 0:
                normalized_df[col] = (comp_df[col] - comp_df[col].min()) / (comp_df[col].max() - comp_df[col].min())
        
        # Select key metrics for visualization
        key_metrics = ['closure_rate', 'median_resolution_days', 'unique_users', 
                      'avg_comments', 'retention_rate', 'has_labels_rate']
        
        # Filter to available metrics
        available_metrics = [m for m in key_metrics if m in normalized_df.columns]
        
        sns.heatmap(normalized_df[available_metrics].T, annot=True, fmt='.2f', 
                   cmap='RdYlGn', center=0.5, ax=ax)
        ax.set_title('Framework Comparison Matrix (Normalized 0-1)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_framework_comparison_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save raw comparison data
        comp_df.to_csv(self.output_dir / 'framework_comparison_data.csv')
        
        return comp_df
        
    def generate_report(self):
        """Generate comprehensive comparison report"""
        report = """# Cross-Framework Comparison Report

## Executive Summary

This report compares issue management patterns across vLLM, SGLang, and llama.cpp frameworks.

## Key Findings

### 1. Basic Metrics
"""
        
        # Add basic metrics comparison
        for framework, metrics in self.results['basic_metrics'].items():
            report += f"\n**{framework}:**\n"
            report += f"- Total Issues: {metrics['total_issues']:,}\n"
            report += f"- Closure Rate: {metrics['closure_rate']:.1%}\n"
            report += f"- Unique Users: {metrics['unique_users']:,}\n"
            report += f"- Median Resolution: {metrics['median_resolution_days']:.1f} days\n"
        
        # Add community health insights
        if 'community_health' in self.results:
            report += "\n### 2. Community Health\n"
            
            retention = self.results['community_health']['retention_metrics']
            best_retention = max(retention.items(), key=lambda x: x[1]['retention_rate'])
            report += f"- Highest retention: {best_retention[0]} ({best_retention[1]['retention_rate']:.1%})\n"
            
            largest_community = max(retention.items(), key=lambda x: x[1]['total_users'])
            report += f"- Largest community: {largest_community[0]} ({largest_community[1]['total_users']:,} users)\n"
        
        # Add issue characteristics
        if 'issue_characteristics' in self.results:
            report += "\n### 3. Issue Characteristics\n"
            
            label_stats = self.results['issue_characteristics']['label_stats']
            most_labels = max(label_stats.items(), key=lambda x: x[1]['unique_labels'])
            report += f"- Most diverse labeling: {most_labels[0]} ({most_labels[1]['unique_labels']} unique labels)\n"
            
            best_labeled = min(label_stats.items(), key=lambda x: x[1]['no_label_rate'])
            report += f"- Best label coverage: {best_labeled[0]} ({(1-best_labeled[1]['no_label_rate'])*100:.1f}% labeled)\n"
        
        report += """
## Recommendations

1. **Best Practices Transfer**: Frameworks can learn from each other's strengths
2. **Community Building**: Focus on user retention and engagement strategies
3. **Process Optimization**: Adopt efficient labeling and resolution practices
4. **Tool Integration**: Consider cross-framework tool compatibility

## Detailed Analysis

See accompanying visualizations for detailed comparisons across multiple dimensions.
"""
        
        return report
        
    def run(self):
        """Run all cross-framework analyses"""
        self.load_all_data()
        self.compare_basic_metrics()
        self.compare_temporal_patterns()
        self.compare_community_health()
        self.compare_issue_characteristics()
        comp_df = self.generate_comparison_matrix()
        
        # Generate report
        report = self.generate_report()
        with open(self.output_dir / 'cross_framework_comparison_report.md', 'w') as f:
            f.write(report)
        
        # Save results
        with open(self.output_dir / 'cross_framework_comparison_data.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nAnalysis complete! Results saved to {self.output_dir}")
        return self.results


if __name__ == "__main__":
    # Define data paths
    data_paths = {
        'vLLM': '/root/yunwei37/vllm-exp/bug-study/data/vllm_all_issues.json',
        'SGLang': '/root/yunwei37/vllm-exp/bug-study/data/sglang_issues.json',
        'llama.cpp': '/root/yunwei37/vllm-exp/bug-study/data/llama_cpp_issues.json'
    }
    
    # Run analysis
    analyzer = CrossFrameworkAnalyzer(
        data_paths,
        '/root/yunwei37/vllm-exp/bug-study/analysis/results/cross_framework'
    )
    analyzer.run()