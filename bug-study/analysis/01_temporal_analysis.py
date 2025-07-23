#!/usr/bin/env python3
"""
Temporal Analysis Module: Analyze time-based patterns in issue data
Research Questions:
- Issue creation patterns over time
- Resolution time distributions
- Activity patterns (hourly, daily, weekly)
- Seasonal trends and anomalies
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TemporalAnalyzer:
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
        self.df['resolution_days'] = pd.NaT
        mask = self.df['is_closed'] & self.df['closed_at'].notna()
        self.df.loc[mask, 'resolution_days'] = (
            self.df.loc[mask, 'closed_at'] - self.df.loc[mask, 'created_at']
        ).dt.total_seconds() / 86400
        
        # Time components
        self.df['created_hour'] = self.df['created_at'].dt.hour
        self.df['created_dow'] = self.df['created_at'].dt.dayofweek
        self.df['created_month'] = self.df['created_at'].dt.month
        self.df['created_year'] = self.df['created_at'].dt.year
        self.df['created_quarter'] = self.df['created_at'].dt.quarter
        
    def analyze_issue_velocity(self):
        """Analyze issue creation and resolution velocity over time"""
        print(f"[{self.framework}] Analyzing issue velocity...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle(f'{self.framework} - Issue Velocity Analysis', fontsize=16, y=1.02)
        
        # 1. Cumulative issues over time
        ax = axes[0, 0]
        daily_created = self.df.groupby(self.df['created_at'].dt.date).size().cumsum()
        daily_closed = self.df[self.df['is_closed']].groupby(self.df[self.df['is_closed']]['closed_at'].dt.date).size().cumsum()
        
        ax.plot(daily_created.index, daily_created.values, label='Created', linewidth=2)
        ax.plot(daily_closed.index, daily_closed.values, label='Closed', linewidth=2)
        ax.fill_between(daily_created.index, daily_created.values, daily_closed.reindex(daily_created.index, fill_value=0).values, 
                       alpha=0.3, label='Open issues')
        ax.set_title('Cumulative Issues Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Issues')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Monthly issue velocity
        ax = axes[0, 1]
        monthly_created = self.df.groupby(pd.Grouper(key='created_at', freq='M')).size()
        monthly_closed = self.df[self.df['is_closed']].groupby(pd.Grouper(key='closed_at', freq='M')).size()
        
        x = monthly_created.index
        width = 10
        ax.bar(x - pd.Timedelta(days=width/2), monthly_created.values, width=width, label='Created', alpha=0.7)
        ax.bar(x + pd.Timedelta(days=width/2), monthly_closed.values, width=width, label='Closed', alpha=0.7)
        ax.set_title('Monthly Issue Volume')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Issues')
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Weekly rolling average
        ax = axes[1, 0]
        daily_new = self.df.groupby(self.df['created_at'].dt.date).size()
        rolling_7d = daily_new.rolling(window=7, min_periods=1).mean()
        rolling_30d = daily_new.rolling(window=30, min_periods=1).mean()
        
        ax.plot(daily_new.index, daily_new.values, alpha=0.3, label='Daily')
        ax.plot(rolling_7d.index, rolling_7d.values, label='7-day avg', linewidth=2)
        ax.plot(rolling_30d.index, rolling_30d.values, label='30-day avg', linewidth=2)
        ax.set_title('Issue Creation Rate (Rolling Averages)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Issues per Day')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Hour of day pattern
        ax = axes[1, 1]
        hourly_counts = self.df['created_hour'].value_counts().sort_index()
        ax.bar(hourly_counts.index, hourly_counts.values, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.set_title('Issues by Hour of Day (UTC)')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Number of Issues')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, axis='y', alpha=0.3)
        
        # 5. Day of week pattern
        ax = axes[2, 0]
        dow_counts = self.df['created_dow'].value_counts().sort_index()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        colors = ['#1f77b4' if i < 5 else '#ff7f0e' for i in range(7)]
        ax.bar(range(7), dow_counts.values, color=colors, alpha=0.7)
        ax.set_xticks(range(7))
        ax.set_xticklabels(dow_names)
        ax.set_title('Issues by Day of Week')
        ax.set_xlabel('Day')
        ax.set_ylabel('Number of Issues')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 6. Quarterly trends
        ax = axes[2, 1]
        quarterly = self.df.groupby(['created_year', 'created_quarter']).size()
        quarters = [f"{y}Q{q}" for (y, q) in quarterly.index]
        ax.plot(range(len(quarterly)), quarterly.values, marker='o', markersize=8, linewidth=2)
        ax.set_xticks(range(len(quarterly)))
        ax.set_xticklabels(quarters, rotation=45, ha='right')
        ax.set_title('Quarterly Issue Trends')
        ax.set_xlabel('Quarter')
        ax.set_ylabel('Number of Issues')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_velocity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['velocity'] = {
            'total_issues': len(self.df),
            'avg_daily_creation': daily_new.mean(),
            'peak_hour': hourly_counts.idxmax(),
            'peak_day': dow_names[dow_counts.idxmax()],
            'weekend_ratio': dow_counts[5:].sum() / dow_counts.sum()
        }
        
    def analyze_resolution_patterns(self):
        """Analyze issue resolution time patterns"""
        print(f"[{self.framework}] Analyzing resolution patterns...")
        
        # Filter for resolved issues
        resolved = self.df[self.df['resolution_days'].notna()].copy()
        if len(resolved) == 0:
            print(f"No resolved issues found for {self.framework}")
            return
        
        # Remove extreme outliers for visualization
        resolved = resolved[resolved['resolution_days'] < resolved['resolution_days'].quantile(0.99)]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.framework} - Resolution Time Analysis', fontsize=16, y=1.02)
        
        # 1. Resolution time distribution
        ax = axes[0, 0]
        ax.hist(resolved['resolution_days'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(resolved['resolution_days'].median(), color='red', linestyle='--', 
                  label=f'Median: {resolved["resolution_days"].median():.1f} days')
        ax.axvline(resolved['resolution_days'].mean(), color='green', linestyle='--', 
                  label=f'Mean: {resolved["resolution_days"].mean():.1f} days')
        ax.set_title('Resolution Time Distribution')
        ax.set_xlabel('Days to Resolution')
        ax.set_ylabel('Number of Issues')
        ax.legend()
        ax.set_xlim(0, resolved['resolution_days'].quantile(0.95))
        
        # 2. Resolution time by month
        ax = axes[0, 1]
        monthly_resolution = resolved.groupby(pd.Grouper(key='created_at', freq='M'))['resolution_days'].agg(['mean', 'median', 'count'])
        monthly_resolution = monthly_resolution[monthly_resolution['count'] >= 5]  # Filter out months with few issues
        
        if len(monthly_resolution) > 0:
            ax.plot(monthly_resolution.index, monthly_resolution['median'], marker='o', label='Median', linewidth=2)
            # Calculate rolling std safely
            rolling_std = monthly_resolution['median'].rolling(3, center=True).std()
            lower_bound = monthly_resolution['median'] - rolling_std
            upper_bound = monthly_resolution['median'] + rolling_std
            # Fill NaN values with the median value to avoid issues
            lower_bound = lower_bound.fillna(monthly_resolution['median'])
            upper_bound = upper_bound.fillna(monthly_resolution['median'])
            ax.fill_between(monthly_resolution.index, 
                           lower_bound,
                           upper_bound,
                           alpha=0.3)
        ax.set_title('Resolution Time Trend')
        ax.set_xlabel('Month')
        ax.set_ylabel('Days to Resolution (Median)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 3. Resolution time by day of week created
        ax = axes[1, 0]
        dow_resolution = resolved.groupby('created_dow')['resolution_days'].median()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        colors = ['#1f77b4' if i < 5 else '#ff7f0e' for i in range(7)]
        ax.bar(range(7), dow_resolution.values, color=colors, alpha=0.7)
        ax.set_xticks(range(7))
        ax.set_xticklabels(dow_names)
        ax.set_title('Median Resolution Time by Day Created')
        ax.set_xlabel('Day Issue Was Created')
        ax.set_ylabel('Days to Resolution (Median)')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 4. Resolution rate over time
        ax = axes[1, 1]
        monthly_stats = self.df.groupby(pd.Grouper(key='created_at', freq='M')).agg({
            'is_closed': ['sum', 'count']
        })
        monthly_stats.columns = ['closed', 'total']
        monthly_stats['resolution_rate'] = monthly_stats['closed'] / monthly_stats['total']
        monthly_stats = monthly_stats[monthly_stats['total'] >= 10]  # Filter out low-volume months
        
        ax.plot(monthly_stats.index, monthly_stats['resolution_rate'], marker='o', linewidth=2)
        ax.set_title('Resolution Rate Over Time')
        ax.set_xlabel('Month')
        ax.set_ylabel('Resolution Rate')
        ax.set_ylim(0, 1.1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_resolution_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['resolution'] = {
            'median_days': resolved['resolution_days'].median(),
            'mean_days': resolved['resolution_days'].mean(),
            'p90_days': resolved['resolution_days'].quantile(0.9),
            'weekend_penalty': (dow_resolution[5:].mean() / dow_resolution[:5].mean()) - 1
        }
        
    def analyze_temporal_anomalies(self):
        """Detect and visualize temporal anomalies"""
        print(f"[{self.framework}] Analyzing temporal anomalies...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.framework} - Temporal Anomaly Detection', fontsize=16, y=1.02)
        
        # 1. Daily issue spikes
        ax = axes[0, 0]
        daily_counts = self.df.groupby(self.df['created_at'].dt.date).size()
        if len(daily_counts) > 30:
            rolling_mean = daily_counts.rolling(window=14, center=True).mean()
            rolling_std = daily_counts.rolling(window=14, center=True).std()
            upper_bound = rolling_mean + 2 * rolling_std
            lower_bound = rolling_mean - 2 * rolling_std
            
            ax.plot(daily_counts.index, daily_counts.values, alpha=0.5, label='Daily count')
            ax.plot(rolling_mean.index, rolling_mean.values, label='14-day average', linewidth=2)
            ax.fill_between(rolling_mean.index, lower_bound, upper_bound, alpha=0.2, label='±2 std dev')
            
            # Mark anomalies
            spikes = daily_counts[daily_counts > upper_bound]
            ax.scatter(spikes.index, spikes.values, color='red', s=50, zorder=5, label=f'Spikes ({len(spikes)})')
            
            ax.set_title('Daily Issue Count Anomalies')
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Issues')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Response time anomalies
        ax = axes[0, 1]
        response_times = []
        for _, row in self.df.iterrows():
            if pd.notna(row['updated_at']) and row['updated_at'] != row['created_at']:
                response_hours = (row['updated_at'] - row['created_at']).total_seconds() / 3600
                if 0 < response_hours < 24*30:  # Reasonable bounds
                    response_times.append((row['created_at'].date(), response_hours))
        
        if response_times:
            response_df = pd.DataFrame(response_times, columns=['date', 'hours'])
            daily_response = response_df.groupby('date')['hours'].median()
            
            if len(daily_response) > 30:
                rolling_median = daily_response.rolling(window=14, center=True).median()
                ax.plot(daily_response.index, daily_response.values, alpha=0.5, label='Daily median')
                ax.plot(rolling_median.index, rolling_median.values, label='14-day median', linewidth=2)
                ax.set_title('Response Time Trends')
                ax.set_xlabel('Date')
                ax.set_ylabel('Hours to First Response (Median)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 3. Seasonal patterns
        ax = axes[1, 0]
        if len(self.df) > 365:  # Need at least a year of data
            monthly_counts = self.df.groupby(self.df['created_at'].dt.month).size()
            ax.bar(monthly_counts.index, monthly_counts.values, color='coral', alpha=0.7)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.set_title('Seasonal Pattern (All Years Combined)')
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Issues')
            ax.grid(True, axis='y', alpha=0.3)
        
        # 4. Hour-day heatmap
        ax = axes[1, 1]
        hour_day_counts = self.df.groupby(['created_dow', 'created_hour']).size().unstack(fill_value=0)
        sns.heatmap(hour_day_counts, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Issue Count'})
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax.set_title('Activity Heatmap (Hour vs Day)')
        ax.set_xlabel('Hour of Day (UTC)')
        ax.set_ylabel('Day of Week')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_temporal_anomalies.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store anomaly results
        if len(daily_counts) > 30:
            self.results['anomalies'] = {
                'spike_dates': spikes.index.tolist() if 'spikes' in locals() else [],
                'spike_count': len(spikes) if 'spikes' in locals() else 0,
                'avg_daily_variance': daily_counts.std() / daily_counts.mean()
            }
    
    def generate_report(self):
        """Generate temporal analysis report"""
        report = f"""# {self.framework} - Temporal Analysis Report

## Issue Velocity
- Total issues: {self.results['velocity']['total_issues']}
- Average daily creation rate: {self.results['velocity']['avg_daily_creation']:.2f}
- Peak activity hour: {self.results['velocity']['peak_hour']}:00 UTC
- Peak activity day: {self.results['velocity']['peak_day']}
- Weekend activity ratio: {self.results['velocity']['weekend_ratio']*100:.1f}%

## Resolution Patterns
"""
        if 'resolution' in self.results:
            res = self.results['resolution']
            report += f"""- Median resolution time: {res['median_days']:.1f} days
- Mean resolution time: {res['mean_days']:.1f} days
- 90th percentile: {res['p90_days']:.1f} days
- Weekend penalty: {res['weekend_penalty']*100:+.1f}% longer resolution
"""
        
        if 'anomalies' in self.results:
            report += f"""
## Temporal Anomalies
- Number of daily spikes: {self.results['anomalies']['spike_count']}
- Daily variance coefficient: {self.results['anomalies']['avg_daily_variance']:.3f}
"""
        
        return report
    
    def run(self):
        """Run all temporal analyses"""
        self.load_data()
        self.analyze_issue_velocity()
        self.analyze_resolution_patterns()
        self.analyze_temporal_anomalies()
        
        # Save report
        report = self.generate_report()
        with open(self.output_dir / f'{self.framework}_temporal_report.md', 'w') as f:
            f.write(report)
        
        # Save data
        with open(self.output_dir / f'{self.framework}_temporal_data.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return self.results


if __name__ == "__main__":
    # Example usage
    analyzer = TemporalAnalyzer(
        'vllm',
        '/root/yunwei37/vllm-exp/bug-study/data/vllm_all_issues.json',
        '/root/yunwei37/vllm-exp/bug-study/analysis/results/temporal'
    )
    analyzer.run()