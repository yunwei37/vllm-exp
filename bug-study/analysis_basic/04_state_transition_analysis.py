#!/usr/bin/env python3
"""
State Transition and Resolution Analysis Module: Analyze issue lifecycle and state transitions
Research Questions:
- Issue state transition patterns
- Resolution success factors
- Time-to-resolution predictors
- Reopened issue patterns
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class StateTransitionAnalyzer:
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
        for col in ['created_at', 'updated_at', 'closed_at']:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Calculate derived fields
        self.df['is_closed'] = self.df['state'] == 'closed'
        self.df['resolution_hours'] = pd.NaT
        mask = self.df['is_closed'] & self.df['closed_at'].notna()
        self.df.loc[mask, 'resolution_hours'] = (
            self.df.loc[mask, 'closed_at'] - self.df.loc[mask, 'created_at']
        ).dt.total_seconds() / 3600
        
        # Response time
        self.df['response_hours'] = (self.df['updated_at'] - self.df['created_at']).dt.total_seconds() / 3600
        self.df.loc[self.df['response_hours'] <= 0, 'response_hours'] = pd.NaT
        
        # Extract additional fields
        self.df['label_count'] = self.df['labels'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        self.df['has_assignee'] = self.df['assignee'].notna()
        self.df['user_login'] = self.df['user'].apply(lambda x: x.get('login') if isinstance(x, dict) else None)
        
    def analyze_state_distributions(self):
        """Analyze issue state and state_reason distributions"""
        print(f"[{self.framework}] Analyzing state distributions...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.framework} - Issue State Analysis', fontsize=16, y=1.02)
        
        # 1. Basic state distribution
        ax = axes[0, 0]
        state_counts = self.df['state'].value_counts()
        colors = ['#2ecc71' if state == 'closed' else '#e74c3c' for state in state_counts.index]
        ax.pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%', colors=colors)
        ax.set_title('Issue State Distribution')
        
        # 2. State reason breakdown
        ax = axes[0, 1]
        closed_df = self.df[self.df['is_closed']]
        if len(closed_df) > 0:
            reason_counts = closed_df['state_reason'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(reason_counts)))
            wedges, texts, autotexts = ax.pie(reason_counts.values, labels=reason_counts.index, 
                                              autopct='%1.1f%%', colors=colors)
            ax.set_title('Closed Issue Reasons')
            plt.setp(autotexts, size=8)
        
        # 3. Resolution time by state reason
        ax = axes[0, 2]
        if 'resolution_hours' in closed_df.columns:
            reason_resolution = closed_df.groupby('state_reason')['resolution_hours'].agg(['median', 'mean', 'count'])
            reason_resolution = reason_resolution[reason_resolution['count'] >= 10]  # Minimum sample size
            
            if len(reason_resolution) > 0:
                x = np.arange(len(reason_resolution))
                ax.bar(x, reason_resolution['median'] / 24, alpha=0.7, label='Median')
                ax.set_xticks(x)
                ax.set_xticklabels(reason_resolution.index, rotation=45, ha='right')
                ax.set_ylabel('Days to Resolution')
                ax.set_title('Resolution Time by Close Reason')
                ax.grid(True, axis='y', alpha=0.3)
                
                # Add count annotations
                for i, (idx, row) in enumerate(reason_resolution.iterrows()):
                    ax.text(i, 1, f'n={int(row["count"])}', ha='center', fontsize=8)
        
        # 4. Monthly closure rate
        ax = axes[1, 0]
        monthly_stats = self.df.groupby(pd.Grouper(key='created_at', freq='M')).agg({
            'is_closed': ['sum', 'count']
        })
        monthly_stats.columns = ['closed', 'total']
        monthly_stats['closure_rate'] = monthly_stats['closed'] / monthly_stats['total']
        monthly_stats = monthly_stats[monthly_stats['total'] >= 10]  # Filter low-volume months
        
        if len(monthly_stats) > 0:
            ax.plot(monthly_stats.index, monthly_stats['closure_rate'], marker='o', linewidth=2)
            ax.fill_between(monthly_stats.index, 0, monthly_stats['closure_rate'], alpha=0.3)
            ax.set_ylabel('Closure Rate')
            ax.set_xlabel('Month')
            ax.set_title('Monthly Closure Rate Trend')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            
            # Add overall average line
            overall_rate = self.df['is_closed'].mean()
            ax.axhline(overall_rate, color='red', linestyle='--', alpha=0.7, 
                      label=f'Overall: {overall_rate:.2%}')
            ax.legend()
        
        # 5. Resolution time distribution
        ax = axes[1, 1]
        resolution_times = self.df['resolution_hours'].dropna()
        if len(resolution_times) > 0:
            # Remove extreme outliers for visualization
            resolution_days = resolution_times / 24
            resolution_days = resolution_days[resolution_days < resolution_days.quantile(0.95)]
            
            ax.hist(resolution_days, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
            
            # Add percentile lines
            percentiles = [50, 75, 90]
            for p in percentiles:
                val = resolution_days.quantile(p/100)
                ax.axvline(val, color='red', linestyle='--', alpha=0.7)
                ax.text(val, ax.get_ylim()[1] * 0.9, f'P{p}: {val:.1f}d', 
                       rotation=90, va='top', ha='right', fontsize=8)
            
            ax.set_xlabel('Days to Resolution')
            ax.set_ylabel('Number of Issues')
            ax.set_title('Resolution Time Distribution (95th percentile)')
            ax.grid(True, axis='y', alpha=0.3)
        
        # 6. Quick vs slow resolution factors
        ax = axes[1, 2]
        if len(resolution_times) > 100:
            # Define quick and slow resolution
            p25 = resolution_times.quantile(0.25)
            p75 = resolution_times.quantile(0.75)
            
            quick_mask = self.df['resolution_hours'] <= p25
            slow_mask = self.df['resolution_hours'] >= p75
            
            quick_stats = {
                'Avg Comments': self.df[quick_mask]['comments'].mean(),
                'Avg Labels': self.df[quick_mask]['label_count'].mean(),
                'Has Assignee %': self.df[quick_mask]['has_assignee'].mean() * 100,
                'Weekend Created %': (self.df[quick_mask]['created_at'].dt.dayofweek >= 5).mean() * 100
            }
            
            slow_stats = {
                'Avg Comments': self.df[slow_mask]['comments'].mean(),
                'Avg Labels': self.df[slow_mask]['label_count'].mean(),
                'Has Assignee %': self.df[slow_mask]['has_assignee'].mean() * 100,
                'Weekend Created %': (self.df[slow_mask]['created_at'].dt.dayofweek >= 5).mean() * 100
            }
            
            metrics = list(quick_stats.keys())
            quick_values = list(quick_stats.values())
            slow_values = list(slow_stats.values())
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax.bar(x - width/2, quick_values, width, label=f'Quick (≤{p25/24:.1f}d)', alpha=0.7, color='green')
            ax.bar(x + width/2, slow_values, width, label=f'Slow (≥{p75/24:.1f}d)', alpha=0.7, color='red')
            
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.set_ylabel('Value')
            ax.set_title('Quick vs Slow Resolution Characteristics')
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_state_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['state_distribution'] = {
            'open_ratio': (~self.df['is_closed']).mean(),
            'closure_rate': self.df['is_closed'].mean(),
            'median_resolution_hours': resolution_times.median() if len(resolution_times) > 0 else None,
            'state_reasons': reason_counts.to_dict() if 'reason_counts' in locals() else {}
        }
        
    def analyze_resolution_factors(self):
        """Analyze factors affecting issue resolution"""
        print(f"[{self.framework}] Analyzing resolution factors...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.framework} - Resolution Factor Analysis', fontsize=16, y=1.02)
        
        # 1. Comment count impact
        ax = axes[0, 0]
        comment_bins = pd.cut(self.df['comments'], bins=[-1, 0, 5, 20, 100, 1000], 
                             labels=['0', '1-5', '6-20', '21-100', '100+'])
        comment_closure = self.df.groupby(comment_bins)['is_closed'].agg(['mean', 'count'])
        comment_closure = comment_closure[comment_closure['count'] >= 10]
        
        if len(comment_closure) > 0:
            ax.bar(range(len(comment_closure)), comment_closure['mean'], alpha=0.7, color='lightblue')
            ax.set_xticks(range(len(comment_closure)))
            ax.set_xticklabels(comment_closure.index)
            ax.set_ylabel('Closure Rate')
            ax.set_xlabel('Number of Comments')
            ax.set_title('Closure Rate by Comment Count')
            ax.set_ylim(0, 1.1)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add count annotations
            for i, (idx, row) in enumerate(comment_closure.iterrows()):
                ax.text(i, 0.05, f'n={int(row["count"])}', ha='center', fontsize=8)
        
        # 2. Label count impact
        ax = axes[0, 1]
        label_closure = self.df.groupby('label_count')['is_closed'].agg(['mean', 'count'])
        label_closure = label_closure[label_closure['count'] >= 20]
        
        if len(label_closure) > 0:
            ax.plot(label_closure.index, label_closure['mean'], marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Labels')
            ax.set_ylabel('Closure Rate')
            ax.set_title('Closure Rate by Label Count')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(label_closure) > 2:
                z = np.polyfit(label_closure.index, label_closure['mean'], 1)
                p = np.poly1d(z)
                ax.plot(label_closure.index, p(label_closure.index), "--", alpha=0.5, color='red')
        
        # 3. Author association impact
        ax = axes[0, 2]
        association_stats = self.df.groupby('author_association').agg({
            'is_closed': ['mean', 'count'],
            'resolution_hours': 'median'
        })
        association_stats.columns = ['closure_rate', 'count', 'median_hours']
        association_stats = association_stats[association_stats['count'] >= 20]
        association_stats = association_stats.sort_values('closure_rate', ascending=False)
        
        if len(association_stats) > 0:
            x = np.arange(len(association_stats))
            ax.bar(x, association_stats['closure_rate'], alpha=0.7, color='lightgreen')
            ax.set_xticks(x)
            ax.set_xticklabels(association_stats.index, rotation=45, ha='right')
            ax.set_ylabel('Closure Rate')
            ax.set_title('Closure Rate by Author Association')
            ax.set_ylim(0, 1.1)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add median resolution time as line
            ax2 = ax.twinx()
            valid_hours = association_stats['median_hours'].dropna()
            if len(valid_hours) > 0:
                ax2.plot(x[:len(valid_hours)], valid_hours / 24, 'ro-', markersize=8)
                ax2.set_ylabel('Median Resolution Days')
        
        # 4. Response time impact
        ax = axes[1, 0]
        response_times = self.df['response_hours'].dropna()
        if len(response_times) > 100:
            # Bin response times
            response_bins = pd.cut(response_times, 
                                 bins=[0, 1, 6, 24, 168, float('inf')],
                                 labels=['<1h', '1-6h', '6-24h', '1-7d', '>7d'])
            
            response_closure = self.df[self.df['response_hours'].notna()].groupby(response_bins).agg({
                'is_closed': 'mean',
                'resolution_hours': 'median',
                'number': 'count'
            })
            response_closure = response_closure[response_closure['number'] >= 20]
            
            if len(response_closure) > 0:
                x = np.arange(len(response_closure))
                ax.bar(x, response_closure['is_closed'], alpha=0.7, color='coral')
                ax.set_xticks(x)
                ax.set_xticklabels(response_closure.index)
                ax.set_ylabel('Closure Rate')
                ax.set_xlabel('First Response Time')
                ax.set_title('Impact of Response Time on Closure')
                ax.set_ylim(0, 1.1)
                ax.grid(True, axis='y', alpha=0.3)
        
        # 5. Weekend vs weekday creation
        ax = axes[1, 1]
        self.df['is_weekend'] = self.df['created_at'].dt.dayofweek >= 5
        weekend_stats = self.df.groupby('is_weekend').agg({
            'is_closed': 'mean',
            'resolution_hours': 'median',
            'comments': 'mean',
            'number': 'count'
        })
        
        metrics = ['Closure Rate', 'Median Resolution (days)', 'Avg Comments']
        weekday_values = [
            weekend_stats.loc[False, 'is_closed'],
            weekend_stats.loc[False, 'resolution_hours'] / 24 if pd.notna(weekend_stats.loc[False, 'resolution_hours']) else 0,
            weekend_stats.loc[False, 'comments']
        ]
        weekend_values = [
            weekend_stats.loc[True, 'is_closed'],
            weekend_stats.loc[True, 'resolution_hours'] / 24 if pd.notna(weekend_stats.loc[True, 'resolution_hours']) else 0,
            weekend_stats.loc[True, 'comments']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, weekday_values, width, label='Weekday', alpha=0.7)
        ax.bar(x + width/2, weekend_values, width, label='Weekend', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Value')
        ax.set_title('Weekday vs Weekend Issue Characteristics')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # 6. Multivariate resolution success
        ax = axes[1, 2]
        # Create success score based on multiple factors
        success_factors = pd.DataFrame({
            'has_labels': self.df['label_count'] > 0,
            'has_comments': self.df['comments'] > 0,
            'quick_response': self.df['response_hours'] < 24,
            'has_assignee': self.df['has_assignee'],
            'is_closed': self.df['is_closed']
        })
        
        # Calculate closure rate for different factor combinations
        factor_impact = {}
        for factor in ['has_labels', 'has_comments', 'quick_response', 'has_assignee']:
            with_factor = success_factors[success_factors[factor]]['is_closed'].mean()
            without_factor = success_factors[~success_factors[factor]]['is_closed'].mean()
            factor_impact[factor] = {
                'with': with_factor,
                'without': without_factor,
                'impact': with_factor - without_factor
            }
        
        factors = list(factor_impact.keys())
        impacts = [factor_impact[f]['impact'] for f in factors]
        colors = ['green' if i > 0 else 'red' for i in impacts]
        
        y_pos = np.arange(len(factors))
        ax.barh(y_pos, impacts, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('_', ' ').title() for f in factors])
        ax.set_xlabel('Impact on Closure Rate')
        ax.set_title('Factor Impact on Issue Closure')
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_resolution_factors.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['resolution_factors'] = {
            'comment_impact': comment_closure['mean'].to_dict() if 'comment_closure' in locals() else {},
            'label_impact': label_closure['mean'].to_dict() if 'label_closure' in locals() else {},
            'weekend_penalty': (weekend_stats.loc[True, 'resolution_hours'] / 
                               weekend_stats.loc[False, 'resolution_hours'] - 1) 
                              if all(pd.notna([weekend_stats.loc[True, 'resolution_hours'],
                                               weekend_stats.loc[False, 'resolution_hours']])) else None,
            'factor_impacts': factor_impact
        }
        
    def analyze_lifecycle_patterns(self):
        """Analyze issue lifecycle patterns and timing"""
        print(f"[{self.framework}] Analyzing lifecycle patterns...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.framework} - Issue Lifecycle Patterns', fontsize=16, y=1.02)
        
        # 1. Time to first response distribution
        ax = axes[0, 0]
        response_hours = self.df['response_hours'].dropna()
        if len(response_hours) > 0:
            # Limit to reasonable range
            response_hours = response_hours[response_hours < 24*30]  # Within 30 days
            
            ax.hist(response_hours, bins=50, edgecolor='black', alpha=0.7, color='lightblue')
            
            # Add percentile markers
            for p in [50, 75, 90]:
                val = response_hours.quantile(p/100)
                ax.axvline(val, color='red', linestyle='--', alpha=0.7)
                label = f'P{p}: {val:.1f}h' if val < 48 else f'P{p}: {val/24:.1f}d'
                ax.text(val, ax.get_ylim()[1] * (0.9 - p/1000), label, 
                       rotation=90, va='top', ha='right', fontsize=8)
            
            ax.set_xlabel('Hours to First Response')
            ax.set_ylabel('Number of Issues')
            ax.set_title('Time to First Response Distribution')
            ax.set_xscale('log')
            ax.grid(True, axis='y', alpha=0.3)
        
        # 2. Issue age distribution (for open issues)
        ax = axes[0, 1]
        open_issues = self.df[~self.df['is_closed']]
        if len(open_issues) > 0:
            current_date = self.df['created_at'].max()
            age_days = (current_date - open_issues['created_at']).dt.days
            
            age_bins = pd.cut(age_days, bins=[0, 7, 30, 90, 365, float('inf')],
                             labels=['<1 week', '1-4 weeks', '1-3 months', '3-12 months', '>1 year'])
            age_dist = age_bins.value_counts()
            
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(age_dist)))
            ax.pie(age_dist.values, labels=[f'{l}\n({v})' for l, v in zip(age_dist.index, age_dist.values)], 
                   autopct='%1.1f%%', colors=colors)
            ax.set_title(f'Age Distribution of {len(open_issues)} Open Issues')
        
        # 3. Resolution speed over issue lifetime
        ax = axes[1, 0]
        # Group by how long the repository has existed
        if len(self.df) > 100:
            self.df['repo_age_at_creation'] = (self.df['created_at'] - self.df['created_at'].min()).dt.days
            
            # Bin by repository age
            age_bins = pd.cut(self.df['repo_age_at_creation'], bins=10)
            resolution_by_age = self.df[self.df['resolution_hours'].notna()].groupby(age_bins).agg({
                'resolution_hours': ['median', 'count'],
                'is_closed': 'mean'
            })
            
            if len(resolution_by_age) > 0:
                resolution_by_age.columns = ['median_hours', 'count', 'closure_rate']
                resolution_by_age = resolution_by_age[resolution_by_age['count'] >= 10]
                
                if len(resolution_by_age) > 0:
                    x = range(len(resolution_by_age))
                    ax.plot(x, resolution_by_age['median_hours'] / 24, marker='o', linewidth=2, label='Median Resolution Days')
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'{int(interval.left)}-{int(interval.right)}d' 
                                       for interval in resolution_by_age.index], rotation=45, ha='right')
                    ax.set_xlabel('Repository Age at Issue Creation')
                    ax.set_ylabel('Median Resolution Days')
                    ax.set_title('Resolution Speed vs Repository Maturity')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
        
        # 4. Issue lifecycle stages
        ax = axes[1, 1]
        # Define lifecycle stages based on various metrics
        lifecycle_stages = {
            'Created': len(self.df),
            'Got Response': (self.df['response_hours'].notna()).sum(),
            'Got Comments': (self.df['comments'] > 0).sum(),
            'Got Labels': (self.df['label_count'] > 0).sum(),
            'Closed': self.df['is_closed'].sum(),
            'Closed Successfully': (self.df['state_reason'] == 'completed').sum()
        }
        
        stages = list(lifecycle_stages.keys())
        counts = list(lifecycle_stages.values())
        
        # Create funnel chart
        y_pos = np.arange(len(stages))
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(stages)))
        
        ax.barh(y_pos, counts, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stages)
        ax.set_xlabel('Number of Issues')
        ax.set_title('Issue Lifecycle Funnel')
        
        # Add percentage annotations
        for i, (stage, count) in enumerate(lifecycle_stages.items()):
            pct = count / lifecycle_stages['Created'] * 100
            ax.text(count + 50, i, f'{count} ({pct:.1f}%)', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_lifecycle_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['lifecycle'] = {
            'median_response_hours': response_hours.median() if 'response_hours' in locals() and len(response_hours) > 0 else None,
            'lifecycle_stages': lifecycle_stages,
            'old_open_issues': (age_days > 365).sum() if 'age_days' in locals() else 0,
            'response_rate': (self.df['response_hours'].notna()).mean()
        }
        
    def analyze_reopened_patterns(self):
        """Analyze patterns in reopened issues (approximated by activity patterns)"""
        print(f"[{self.framework}] Analyzing reopened issue patterns...")
        
        # Since we don't have explicit reopen events, we'll look for patterns that suggest reopening
        # such as issues with high comment counts after being closed, or specific state_reasons
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.framework} - Issue Reopening Patterns (Approximated)', fontsize=16, y=1.02)
        
        # 1. State reason analysis for potential reopens
        ax = axes[0, 0]
        state_reasons = self.df['state_reason'].value_counts()
        
        # Identify potential reopen indicators
        reopen_indicators = ['reopened', 'not_planned', 'duplicate']
        potential_reopens = state_reasons[state_reasons.index.isin(reopen_indicators)]
        other_reasons = state_reasons[~state_reasons.index.isin(reopen_indicators)]
        
        if len(potential_reopens) > 0 or len(other_reasons) > 0:
            labels = list(potential_reopens.index) + ['other']
            values = list(potential_reopens.values) + [other_reasons.sum()]
            colors = ['red'] * len(potential_reopens) + ['gray']
            
            ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors)
            ax.set_title('Potential Reopen Indicators in State Reasons')
        
        # 2. High activity closed issues
        ax = axes[0, 1]
        closed_issues = self.df[self.df['is_closed']]
        if len(closed_issues) > 0:
            # Look for closed issues with unusually high comment counts
            high_activity_threshold = closed_issues['comments'].quantile(0.9)
            high_activity_closed = closed_issues[closed_issues['comments'] > high_activity_threshold]
            
            comment_bins = pd.cut(closed_issues['comments'], 
                                 bins=[-1, 5, 20, 50, 100, 1000],
                                 labels=['0-5', '6-20', '21-50', '51-100', '100+'])
            comment_dist = comment_bins.value_counts()
            
            ax.bar(comment_dist.index, comment_dist.values, color='lightblue', alpha=0.7)
            ax.set_xlabel('Comment Count')
            ax.set_ylabel('Number of Closed Issues')
            ax.set_title('Comment Distribution in Closed Issues')
            ax.grid(True, axis='y', alpha=0.3)
            
            # Highlight high activity region
            if high_activity_threshold > 0:
                ax.axvline(high_activity_threshold, color='red', linestyle='--', 
                          label=f'High activity (>{high_activity_threshold:.0f})')
                ax.legend()
        
        # 3. Quick closure analysis
        ax = axes[1, 0]
        if 'resolution_hours' in self.df.columns:
            quick_closed = self.df[self.df['resolution_hours'] < 24]  # Closed within 24 hours
            
            if len(quick_closed) > 0:
                # Analyze characteristics of quickly closed issues
                quick_stats = {
                    'No Comments': (quick_closed['comments'] == 0).mean() * 100,
                    'Not Planned': (quick_closed['state_reason'] == 'not_planned').mean() * 100,
                    'Duplicate': (quick_closed['state_reason'] == 'duplicate').mean() * 100,
                    'Completed': (quick_closed['state_reason'] == 'completed').mean() * 100
                }
                
                metrics = list(quick_stats.keys())
                values = list(quick_stats.values())
                
                ax.bar(metrics, values, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
                ax.set_ylabel('Percentage')
                ax.set_title(f'Characteristics of {len(quick_closed)} Quick-Closed Issues (<24h)')
                ax.grid(True, axis='y', alpha=0.3)
        
        # 4. Resolution time by author association (potential reopen predictor)
        ax = axes[1, 1]
        # Theory: issues from certain author types might be more likely to be reopened
        association_resolution = self.df.groupby('author_association').agg({
            'resolution_hours': 'median',
            'state_reason': lambda x: (x == 'not_planned').mean(),
            'comments': 'mean',
            'number': 'count'
        })
        association_resolution.columns = ['median_hours', 'not_planned_rate', 'avg_comments', 'count']
        association_resolution = association_resolution[association_resolution['count'] >= 20]
        
        if len(association_resolution) > 0:
            x = np.arange(len(association_resolution))
            width = 0.35
            
            ax.bar(x - width/2, association_resolution['not_planned_rate'] * 100, width, 
                   label='Not Planned %', alpha=0.7, color='red')
            ax.bar(x + width/2, association_resolution['avg_comments'], width, 
                   label='Avg Comments', alpha=0.7, color='blue')
            
            ax.set_xticks(x)
            ax.set_xticklabels(association_resolution.index, rotation=45, ha='right')
            ax.set_ylabel('Value')
            ax.set_title('Potential Reopen Indicators by Author Type')
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_reopened_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['reopen_patterns'] = {
            'not_planned_ratio': (self.df['state_reason'] == 'not_planned').mean(),
            'duplicate_ratio': (self.df['state_reason'] == 'duplicate').mean(),
            'quick_closure_ratio': (self.df['resolution_hours'] < 24).sum() / len(self.df[self.df['is_closed']]) 
                                  if len(self.df[self.df['is_closed']]) > 0 else 0,
            'high_activity_closed_count': len(high_activity_closed) if 'high_activity_closed' in locals() else 0
        }
    
    def generate_report(self):
        """Generate state transition analysis report"""
        report = f"""# {self.framework} - State Transition and Resolution Analysis Report

## State Distribution
- Open issues: {self.results['state_distribution']['open_ratio']*100:.1f}%
- Closed issues: {self.results['state_distribution']['closure_rate']*100:.1f}%
- Median resolution time: {self.results['state_distribution']['median_resolution_hours']/24:.1f} days

### Closure Reasons:
"""
        for reason, count in self.results['state_distribution']['state_reasons'].items():
            report += f"- {reason}: {count}\n"
        
        report += f"""
## Resolution Factors
"""
        if 'resolution_factors' in self.results:
            report += f"- Weekend resolution penalty: {self.results['resolution_factors']['weekend_penalty']*100:+.1f}%\n"
            
            report += "\n### Factor Impacts on Closure:\n"
            for factor, impact in self.results['resolution_factors']['factor_impacts'].items():
                report += f"- {factor.replace('_', ' ').title()}: {impact['impact']*100:+.1f}%\n"
        
        if 'lifecycle' in self.results:
            report += f"""
## Lifecycle Patterns
- Median response time: {self.results['lifecycle']['median_response_hours']:.1f} hours
- Response rate: {self.results['lifecycle']['response_rate']*100:.1f}%
- Old open issues (>1 year): {self.results['lifecycle']['old_open_issues']}

### Lifecycle Funnel:
"""
            total = self.results['lifecycle']['lifecycle_stages']['Created']
            for stage, count in self.results['lifecycle']['lifecycle_stages'].items():
                report += f"- {stage}: {count} ({count/total*100:.1f}%)\n"
        
        if 'reopen_patterns' in self.results:
            report += f"""
## Reopening Patterns (Approximated)
- Not planned ratio: {self.results['reopen_patterns']['not_planned_ratio']*100:.1f}%
- Duplicate ratio: {self.results['reopen_patterns']['duplicate_ratio']*100:.1f}%
- Quick closure ratio (<24h): {self.results['reopen_patterns']['quick_closure_ratio']*100:.1f}%
- High activity closed issues: {self.results['reopen_patterns']['high_activity_closed_count']}
"""
        
        return report
    
    def run(self):
        """Run all state transition analyses"""
        self.load_data()
        self.analyze_state_distributions()
        self.analyze_resolution_factors()
        self.analyze_lifecycle_patterns()
        self.analyze_reopened_patterns()
        
        # Save report
        report = self.generate_report()
        with open(self.output_dir / f'{self.framework}_state_transition_report.md', 'w') as f:
            f.write(report)
        
        # Save data
        with open(self.output_dir / f'{self.framework}_state_transition_data.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return self.results


if __name__ == "__main__":
    # Example usage
    analyzer = StateTransitionAnalyzer(
        'vllm',
        '/root/yunwei37/vllm-exp/bug-study/data/vllm_all_issues.json',
        '/root/yunwei37/vllm-exp/bug-study/analysis/results/state_transition'
    )
    analyzer.run()