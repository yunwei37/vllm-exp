#!/usr/bin/env python3
"""
Advanced Label Analysis for Debugging and Performance
Implements the research questions from Section 9-14 of README.md
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class AdvancedLabelAnalyzer:
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
        self.df['updated_at'] = pd.to_datetime(self.df['updated_at'], errors='coerce')
        
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
        
        # Time to first update (proxy for labeling time)
        self.df['time_to_update'] = (self.df['updated_at'] - self.df['created_at']).dt.total_seconds() / 3600  # hours
        
        print(f"Loaded {len(self.df)} issues for {self.framework}")
        
    def analyze_label_anomalies(self):
        """Section 9: Label Anomaly Detection"""
        print(f"[{self.framework}] Analyzing label anomalies...")
        
        # Get all labels and their usage over time
        all_labels = []
        for labels in self.df['label_names']:
            all_labels.extend(labels)
        label_counts = Counter(all_labels)
        
        # Calculate usage in last 6 months
        six_months_ago = self.df['created_at'].max() - timedelta(days=180)
        recent_df = self.df[self.df['created_at'] >= six_months_ago]
        
        recent_labels = []
        for labels in recent_df['label_names']:
            recent_labels.extend(labels)
        recent_label_counts = Counter(recent_labels)
        
        # Find zombie labels (used < 5 times in last 6 months)
        zombie_labels = []
        for label, total_count in label_counts.items():
            recent_count = recent_label_counts.get(label, 0)
            if recent_count < 5 and total_count >= 10:  # Was used before but not recently
                # Find last usage
                last_usage = None
                for _, row in self.df.sort_values('created_at', ascending=False).iterrows():
                    if label in row['label_names']:
                        last_usage = row['created_at']
                        break
                zombie_labels.append({
                    'label': label,
                    'total_count': total_count,
                    'recent_count': recent_count,
                    'last_used': last_usage
                })
        
        # Find orphaned labels (only on closed issues)
        orphaned_labels = []
        for label in label_counts:
            open_mask = self.df['label_names'].apply(lambda x: label in x) & ~self.df['is_closed']
            closed_mask = self.df['label_names'].apply(lambda x: label in x) & self.df['is_closed']
            
            if open_mask.sum() == 0 and closed_mask.sum() > 0:
                orphaned_labels.append({
                    'label': label,
                    'closed_count': closed_mask.sum()
                })
        
        # Find solo labels (never co-occur)
        solo_labels = []
        for label in label_counts:
            if label_counts[label] >= 5:  # Minimum threshold
                co_occurrences = 0
                solo_occurrences = 0
                for labels in self.df['label_names']:
                    if label in labels:
                        if len(labels) == 1:
                            solo_occurrences += 1
                        else:
                            co_occurrences += 1
                
                if co_occurrences == 0 and solo_occurrences > 0:
                    solo_labels.append({
                        'label': label,
                        'count': solo_occurrences
                    })
        
        # Detect label usage spikes
        label_spikes = self._detect_label_spikes()
        
        # Store results
        self.results['anomalies'] = {
            'zombie_labels': len(zombie_labels),
            'orphaned_labels': len(orphaned_labels),
            'solo_labels': len(solo_labels),
            'label_spikes': len(label_spikes),
            'zombie_details': zombie_labels[:10],  # Top 10
            'orphaned_details': orphaned_labels[:10],
            'solo_details': solo_labels[:10]
        }
        
        # Visualize
        self._visualize_anomalies(zombie_labels, orphaned_labels, solo_labels, label_spikes)
        
    def _detect_label_spikes(self):
        """Detect sudden increases in label usage"""
        spikes = []
        
        # Get top labels
        all_labels = []
        for labels in self.df['label_names']:
            all_labels.extend(labels)
        label_counts = Counter(all_labels)
        
        for label, count in label_counts.most_common(30):  # Top 30 labels
            if count < 20:  # Need minimum data
                continue
                
            # Calculate weekly usage
            weekly_usage = defaultdict(int)
            for _, row in self.df.iterrows():
                if label in row['label_names']:
                    week = row['created_at'].isocalendar()[:2]  # (year, week)
                    weekly_usage[week] += 1
            
            if len(weekly_usage) < 10:  # Need enough weeks
                continue
                
            # Calculate z-scores
            usage_values = list(weekly_usage.values())
            mean_usage = np.mean(usage_values)
            std_usage = np.std(usage_values)
            
            if std_usage > 0:
                for week, usage in weekly_usage.items():
                    z_score = (usage - mean_usage) / std_usage
                    if z_score > 3:  # 3 standard deviations
                        spikes.append({
                            'label': label,
                            'week': week,
                            'usage': usage,
                            'z_score': z_score,
                            'mean_usage': mean_usage
                        })
        
        return spikes
        
    def _visualize_anomalies(self, zombie_labels, orphaned_labels, solo_labels, label_spikes):
        """Visualize label anomalies"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.framework} - Label Anomaly Detection', fontsize=16, y=1.02)
        
        # 1. Zombie labels
        ax = axes[0, 0]
        if zombie_labels:
            zombie_df = pd.DataFrame(zombie_labels[:10])
            y_pos = np.arange(len(zombie_df))
            bars = ax.barh(y_pos, zombie_df['total_count'], color='red', alpha=0.7)
            ax.barh(y_pos, zombie_df['recent_count'], color='green', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(zombie_df['label'], fontsize=8)
            ax.set_xlabel('Usage Count')
            ax.set_title('Zombie Labels (Red=Total, Green=Recent 6mo)')
            ax.grid(True, axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No zombie labels found', ha='center', va='center')
        
        # 2. Orphaned labels
        ax = axes[0, 1]
        if orphaned_labels:
            orphan_df = pd.DataFrame(orphaned_labels[:15])
            ax.bar(range(len(orphan_df)), orphan_df['closed_count'], color='orange', alpha=0.7)
            ax.set_xticks(range(len(orphan_df)))
            ax.set_xticklabels(orphan_df['label'], rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Count (Closed Issues Only)')
            ax.set_title('Orphaned Labels (Only on Closed Issues)')
            ax.grid(True, axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No orphaned labels found', ha='center', va='center')
        
        # 3. Solo labels
        ax = axes[1, 0]
        if solo_labels:
            solo_df = pd.DataFrame(solo_labels[:10])
            ax.pie(solo_df['count'], labels=solo_df['label'], autopct='%1.0f', startangle=90)
            ax.set_title('Solo Labels (Never Co-occur)')
        else:
            ax.text(0.5, 0.5, 'No solo labels found', ha='center', va='center')
        
        # 4. Label spikes timeline
        ax = axes[1, 1]
        if label_spikes:
            # Group spikes by label
            spike_labels = defaultdict(list)
            for spike in label_spikes[:20]:  # Top 20 spikes
                spike_labels[spike['label']].append(spike)
            
            # Plot top 5 labels with spikes
            colors = plt.cm.Set3(np.linspace(0, 1, min(5, len(spike_labels))))
            for i, (label, spikes) in enumerate(list(spike_labels.items())[:5]):
                weeks = [s['week'][1] for s in spikes]  # Week number
                z_scores = [s['z_score'] for s in spikes]
                ax.scatter(weeks, z_scores, label=label, color=colors[i], s=100, alpha=0.7)
            
            ax.axhline(3, color='red', linestyle='--', alpha=0.5, label='3σ threshold')
            ax.set_xlabel('Week of Year')
            ax.set_ylabel('Z-Score')
            ax.set_title('Label Usage Spikes')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No significant spikes found', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_label_anomalies.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_label_quality(self):
        """Section 10: Label Quality and Consistency"""
        print(f"[{self.framework}] Analyzing label quality...")
        
        # Labeling delay analysis
        labeling_delays = []
        for _, row in self.df.iterrows():
            if row['label_count'] > 0 and pd.notna(row['time_to_update']):
                # Use update time as proxy for labeling time
                if row['time_to_update'] >= 0 and row['time_to_update'] < 24*30:  # Within 30 days
                    labeling_delays.append(row['time_to_update'])
        
        # Label coverage
        coverage_stats = {
            'no_labels': (self.df['label_count'] == 0).sum(),
            'one_label': (self.df['label_count'] == 1).sum(),
            'multi_labels': (self.df['label_count'] > 1).sum(),
            'avg_labels': self.df['label_count'].mean(),
            'median_labels': self.df['label_count'].median()
        }
        
        # Label stability (for this we need to track changes - using proxies)
        # High comment count with few labels might indicate label changes
        stability_proxy = self.df[self.df['comments'] > 5].copy()
        stability_proxy['instability_score'] = stability_proxy['comments'] / (stability_proxy['label_count'] + 1)
        
        # Store results
        self.results['quality'] = {
            'avg_labeling_delay_hours': np.mean(labeling_delays) if labeling_delays else None,
            'median_labeling_delay_hours': np.median(labeling_delays) if labeling_delays else None,
            'no_label_ratio': coverage_stats['no_labels'] / len(self.df),
            'single_label_ratio': coverage_stats['one_label'] / len(self.df),
            'multi_label_ratio': coverage_stats['multi_labels'] / len(self.df),
            'avg_labels_per_issue': coverage_stats['avg_labels'],
            'high_instability_issues': (stability_proxy['instability_score'] > 10).sum()
        }
        
        # Visualize
        self._visualize_quality(labeling_delays, coverage_stats, stability_proxy)
        
    def _visualize_quality(self, labeling_delays, coverage_stats, stability_proxy):
        """Visualize label quality metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.framework} - Label Quality Analysis', fontsize=16, y=1.02)
        
        # 1. Labeling delay distribution
        ax = axes[0, 0]
        if labeling_delays:
            ax.hist(labeling_delays, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(labeling_delays), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(labeling_delays):.1f}h')
            ax.axvline(np.median(labeling_delays), color='green', linestyle='--', 
                      label=f'Median: {np.median(labeling_delays):.1f}h')
            ax.set_xlabel('Hours to First Update')
            ax.set_ylabel('Number of Issues')
            ax.set_title('Labeling Delay Distribution')
            ax.set_xlim(0, min(100, max(labeling_delays)))
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
        
        # 2. Label coverage pie chart
        ax = axes[0, 1]
        sizes = [coverage_stats['no_labels'], coverage_stats['one_label'], coverage_stats['multi_labels']]
        labels = ['No Labels', 'Single Label', 'Multiple Labels']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Label Coverage Distribution')
        
        # 3. Label count distribution
        ax = axes[1, 0]
        label_count_dist = self.df['label_count'].value_counts().sort_index()
        ax.bar(label_count_dist.index[:10], label_count_dist.values[:10], color='skyblue', alpha=0.7)
        ax.set_xlabel('Number of Labels')
        ax.set_ylabel('Number of Issues')
        ax.set_title('Label Count Distribution (up to 10)')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 4. Instability score distribution
        ax = axes[1, 1]
        if len(stability_proxy) > 0:
            ax.scatter(stability_proxy['label_count'], stability_proxy['instability_score'], 
                      alpha=0.5, s=20)
            ax.set_xlabel('Number of Labels')
            ax.set_ylabel('Instability Score (Comments/Labels)')
            ax.set_title('Label Stability Analysis')
            ax.set_ylim(0, min(50, stability_proxy['instability_score'].max()))
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_label_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_performance_bottlenecks(self):
        """Section 11: Performance Bottleneck Identification"""
        print(f"[{self.framework}] Analyzing performance bottlenecks...")
        
        # Get all labels
        all_labels = []
        for labels in self.df['label_names']:
            all_labels.extend(labels)
        label_counts = Counter(all_labels)
        
        # Analyze stalled patterns
        stalled_patterns = {}
        for label, count in label_counts.items():
            if count >= 20:  # Minimum threshold
                mask = self.df['label_names'].apply(lambda x: label in x)
                label_issues = self.df[mask]
                
                # Calculate resolution metrics
                open_issues = label_issues[~label_issues['is_closed']]
                resolved_issues = label_issues[label_issues['resolution_days'].notna()]
                
                if len(resolved_issues) > 5:
                    stalled_patterns[label] = {
                        'total': len(label_issues),
                        'open': len(open_issues),
                        'open_ratio': len(open_issues) / len(label_issues),
                        'median_resolution': resolved_issues['resolution_days'].median(),
                        'p90_resolution': resolved_issues['resolution_days'].quantile(0.9),
                        'p99_resolution': resolved_issues['resolution_days'].quantile(0.99),
                        'avg_comments': label_issues['comments'].mean()
                    }
        
        # Find comment explosion labels
        comment_explosion = {}
        for label, count in label_counts.items():
            if count >= 15:
                mask = self.df['label_names'].apply(lambda x: label in x)
                with_label = self.df[mask]['comments'].mean()
                without_label = self.df[~mask]['comments'].mean()
                
                if with_label > without_label * 1.5:  # 50% more comments
                    comment_explosion[label] = {
                        'avg_comments_with': with_label,
                        'avg_comments_without': without_label,
                        'ratio': with_label / (without_label + 0.1),
                        'count': count
                    }
        
        # Analyze label combinations for stalled issues
        long_open = self.df[~self.df['is_closed'] & 
                           (self.df['created_at'] < self.df['created_at'].max() - timedelta(days=90))]
        
        combo_patterns = Counter()
        for labels in long_open['label_names']:
            if len(labels) >= 2:
                for combo in combinations(sorted(labels), 2):
                    combo_patterns[combo] += 1
        
        # Store results
        self.results['bottlenecks'] = {
            'high_stall_labels': len([l for l, v in stalled_patterns.items() if v['open_ratio'] > 0.5]),
            'comment_explosion_labels': len(comment_explosion),
            'problematic_combinations': len([c for c, v in combo_patterns.items() if v >= 5]),
            'worst_stall_label': max(stalled_patterns.items(), key=lambda x: x[1]['open_ratio'])[0] if stalled_patterns else None,
            'worst_comment_label': max(comment_explosion.items(), key=lambda x: x[1]['ratio'])[0] if comment_explosion else None
        }
        
        # Visualize
        self._visualize_bottlenecks(stalled_patterns, comment_explosion, combo_patterns)
        
    def _visualize_bottlenecks(self, stalled_patterns, comment_explosion, combo_patterns):
        """Visualize performance bottlenecks"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.framework} - Performance Bottleneck Analysis', fontsize=16, y=1.02)
        
        # 1. Stalled issue patterns
        ax = axes[0, 0]
        if stalled_patterns:
            # Sort by open ratio
            sorted_stalled = sorted(stalled_patterns.items(), key=lambda x: x[1]['open_ratio'], reverse=True)[:10]
            labels = [x[0] for x in sorted_stalled]
            open_ratios = [x[1]['open_ratio'] for x in sorted_stalled]
            
            y_pos = np.arange(len(labels))
            bars = ax.barh(y_pos, open_ratios, color='red', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Open Issue Ratio')
            ax.set_title('Labels with Highest Stall Rates')
            ax.set_xlim(0, 1)
            ax.grid(True, axis='x', alpha=0.3)
            
            # Add counts
            for i, (label, data) in enumerate(sorted_stalled):
                ax.text(0.02, i, f"n={data['total']}", va='center', fontsize=7)
        
        # 2. Resolution time by label
        ax = axes[0, 1]
        if stalled_patterns:
            # Sort by P90 resolution time
            valid_patterns = {k: v for k, v in stalled_patterns.items() if pd.notna(v['p90_resolution'])}
            sorted_resolution = sorted(valid_patterns.items(), key=lambda x: x[1]['p90_resolution'], reverse=True)[:10]
            
            if sorted_resolution:
                labels = [x[0] for x in sorted_resolution]
                medians = [x[1]['median_resolution'] for x in sorted_resolution]
                p90s = [x[1]['p90_resolution'] for x in sorted_resolution]
                
                x = np.arange(len(labels))
                width = 0.35
                
                ax.bar(x - width/2, medians, width, label='Median', alpha=0.7)
                ax.bar(x + width/2, p90s, width, label='P90', alpha=0.7)
                
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Resolution Time (days)')
                ax.set_title('Resolution Time by Label')
                ax.legend()
                ax.grid(True, axis='y', alpha=0.3)
        
        # 3. Comment explosion
        ax = axes[1, 0]
        if comment_explosion:
            sorted_explosion = sorted(comment_explosion.items(), key=lambda x: x[1]['ratio'], reverse=True)[:10]
            labels = [x[0] for x in sorted_explosion]
            ratios = [x[1]['ratio'] for x in sorted_explosion]
            
            ax.bar(range(len(labels)), ratios, color='orange', alpha=0.7)
            ax.axhline(1, color='black', linestyle='--', alpha=0.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Comment Ratio (vs avg)')
            ax.set_title('Labels with Comment Explosion')
            ax.grid(True, axis='y', alpha=0.3)
        
        # 4. Problematic label combinations
        ax = axes[1, 1]
        if combo_patterns:
            top_combos = combo_patterns.most_common(10)
            combo_labels = [f"{c[0][0]}\n+\n{c[0][1]}" for c in top_combos]
            combo_counts = [c[1] for c in top_combos]
            
            y_pos = np.arange(len(top_combos))
            ax.barh(y_pos, combo_counts, color='purple', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(combo_labels, fontsize=7)
            ax.set_xlabel('Long-Open Issues')
            ax.set_title('Label Combinations in Stalled Issues')
            ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_performance_bottlenecks.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_label_optimization(self):
        """Section 12: Label Optimization and Redundancy"""
        print(f"[{self.framework}] Analyzing label optimization...")
        
        # Calculate co-occurrence matrix
        all_labels = []
        for labels in self.df['label_names']:
            all_labels.extend(labels)
        label_counts = Counter(all_labels)
        
        # Get significant labels
        significant_labels = [label for label, count in label_counts.items() if count >= 20]
        
        # Build co-occurrence matrix
        cooccurrence = defaultdict(lambda: defaultdict(int))
        for labels in self.df['label_names']:
            for label1 in labels:
                for label2 in labels:
                    if label1 != label2 and label1 in significant_labels and label2 in significant_labels:
                        cooccurrence[label1][label2] += 1
        
        # Find redundant labels (high co-occurrence)
        redundant_pairs = []
        for label1 in significant_labels:
            for label2 in significant_labels:
                if label1 < label2:  # Avoid duplicates
                    count1 = label_counts[label1]
                    count2 = label_counts[label2]
                    co_count = cooccurrence[label1][label2]
                    
                    if co_count > 0:
                        jaccard = co_count / (count1 + count2 - co_count)
                        confidence1 = co_count / count1
                        confidence2 = co_count / count2
                        
                        if jaccard > 0.5 or confidence1 > 0.8 or confidence2 > 0.8:
                            redundant_pairs.append({
                                'label1': label1,
                                'label2': label2,
                                'jaccard': jaccard,
                                'confidence1': confidence1,
                                'confidence2': confidence2,
                                'co_count': co_count
                            })
        
        # Minimal label set analysis
        label_coverage = self._find_minimal_label_set(significant_labels)
        
        # Store results
        self.results['optimization'] = {
            'redundant_pairs': len(redundant_pairs),
            'highest_redundancy': max(redundant_pairs, key=lambda x: x['jaccard']) if redundant_pairs else None,
            'minimal_set_50': label_coverage['labels_for_50'],
            'minimal_set_80': label_coverage['labels_for_80'],
            'minimal_set_95': label_coverage['labels_for_95'],
            'total_labels': len(label_counts)
        }
        
        # Visualize
        self._visualize_optimization(redundant_pairs, label_coverage, cooccurrence, significant_labels)
        
    def _find_minimal_label_set(self, significant_labels):
        """Find minimal set of labels covering most issues"""
        # Count issues covered by each label
        label_coverage_counts = {}
        for label in significant_labels:
            mask = self.df['label_names'].apply(lambda x: label in x)
            label_coverage_counts[label] = mask.sum()
        
        # Greedy set cover
        covered_issues = set()
        selected_labels = []
        total_labeled = (self.df['label_count'] > 0).sum()
        
        coverage_milestones = {'labels_for_50': 0, 'labels_for_80': 0, 'labels_for_95': 0}
        
        while len(covered_issues) < total_labeled and label_coverage_counts:
            # Find label covering most uncovered issues
            best_label = None
            best_new_coverage = 0
            
            for label, _ in sorted(label_coverage_counts.items(), key=lambda x: x[1], reverse=True):
                mask = self.df['label_names'].apply(lambda x: label in x)
                issue_indices = set(self.df[mask].index)
                new_coverage = len(issue_indices - covered_issues)
                
                if new_coverage > best_new_coverage:
                    best_label = label
                    best_new_coverage = new_coverage
            
            if best_label:
                selected_labels.append(best_label)
                mask = self.df['label_names'].apply(lambda x: best_label in x)
                covered_issues.update(self.df[mask].index)
                del label_coverage_counts[best_label]
                
                # Check milestones
                coverage_pct = len(covered_issues) / total_labeled
                if coverage_pct >= 0.5 and coverage_milestones['labels_for_50'] == 0:
                    coverage_milestones['labels_for_50'] = len(selected_labels)
                if coverage_pct >= 0.8 and coverage_milestones['labels_for_80'] == 0:
                    coverage_milestones['labels_for_80'] = len(selected_labels)
                if coverage_pct >= 0.95 and coverage_milestones['labels_for_95'] == 0:
                    coverage_milestones['labels_for_95'] = len(selected_labels)
            else:
                break
        
        return coverage_milestones
        
    def _visualize_optimization(self, redundant_pairs, label_coverage, cooccurrence, significant_labels):
        """Visualize label optimization opportunities"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.framework} - Label Optimization Analysis', fontsize=16, y=1.02)
        
        # 1. Redundant label pairs
        ax = axes[0, 0]
        if redundant_pairs:
            sorted_pairs = sorted(redundant_pairs, key=lambda x: x['jaccard'], reverse=True)[:10]
            pair_labels = [f"{p['label1']}\nvs\n{p['label2']}" for p in sorted_pairs]
            jaccards = [p['jaccard'] for p in sorted_pairs]
            
            y_pos = np.arange(len(sorted_pairs))
            ax.barh(y_pos, jaccards, color='red', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pair_labels, fontsize=7)
            ax.set_xlabel('Jaccard Similarity')
            ax.set_title('Most Redundant Label Pairs')
            ax.set_xlim(0, 1)
            ax.grid(True, axis='x', alpha=0.3)
        
        # 2. Label coverage efficiency
        ax = axes[0, 1]
        coverage_data = [
            ('50% coverage', label_coverage['labels_for_50']),
            ('80% coverage', label_coverage['labels_for_80']),
            ('95% coverage', label_coverage['labels_for_95']),
            ('Total labels', len(significant_labels))
        ]
        
        labels = [x[0] for x in coverage_data]
        values = [x[1] for x in coverage_data]
        
        bars = ax.bar(range(len(labels)), values, color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Number of Labels')
        ax.set_title('Minimal Label Set Analysis')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add percentage annotations
        for i, (label, value) in enumerate(coverage_data[:-1]):
            pct = value / values[-1] * 100
            ax.text(i, value + 1, f'{pct:.0f}%', ha='center', fontsize=9)
        
        # 3. Co-occurrence heatmap (top labels)
        ax = axes[1, 0]
        top_labels = sorted(significant_labels, key=lambda x: sum(cooccurrence[x].values()), reverse=True)[:15]
        
        matrix = np.zeros((len(top_labels), len(top_labels)))
        for i, label1 in enumerate(top_labels):
            for j, label2 in enumerate(top_labels):
                if i != j:
                    matrix[i, j] = cooccurrence[label1][label2]
        
        # Normalize by diagonal
        for i in range(len(top_labels)):
            if matrix[i, i] > 0:
                matrix[i, :] = matrix[i, :] / (matrix[i, i] + 1)
                matrix[:, i] = matrix[:, i] / (matrix[i, i] + 1)
        
        sns.heatmap(matrix, xticklabels=top_labels, yticklabels=top_labels,
                   cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Co-occurrence Rate'})
        ax.set_title('Label Co-occurrence Matrix')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
        
        # 4. Label efficiency scatter
        ax = axes[1, 1]
        # Plot label frequency vs unique coverage
        label_efficiency = []
        for label in significant_labels[:30]:
            mask = self.df['label_names'].apply(lambda x: label in x)
            total_issues = mask.sum()
            
            # How many issues have only this label
            solo_mask = mask & (self.df['label_count'] == 1)
            solo_issues = solo_mask.sum()
            
            if total_issues > 0:
                label_efficiency.append({
                    'label': label,
                    'total': total_issues,
                    'solo': solo_issues,
                    'efficiency': solo_issues / total_issues
                })
        
        if label_efficiency:
            eff_df = pd.DataFrame(label_efficiency)
            scatter = ax.scatter(eff_df['total'], eff_df['efficiency'], 
                               s=eff_df['solo']*2, alpha=0.6, c=eff_df['efficiency'], cmap='viridis')
            
            # Annotate interesting points
            for _, row in eff_df.iterrows():
                if row['efficiency'] > 0.5 or row['total'] > 100:
                    ax.annotate(row['label'], (row['total'], row['efficiency']), 
                               fontsize=7, alpha=0.7)
            
            ax.set_xlabel('Total Usage Count')
            ax.set_ylabel('Solo Usage Rate')
            ax.set_title('Label Efficiency (Size = Solo Count)')
            ax.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Efficiency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_label_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_complete_distribution(self):
        """Section 13: Complete Label Distribution Analysis"""
        print(f"[{self.framework}] Analyzing complete label distribution...")
        
        # Get ALL labels
        all_labels = []
        for labels in self.df['label_names']:
            all_labels.extend(labels)
        label_counts = Counter(all_labels)
        
        # Create complete distribution dataframe
        distribution_df = pd.DataFrame([
            {
                'label': label,
                'count': count,
                'percentage': count / len(self.df) * 100,
                'issues_percentage': count / len(self.df) * 100
            }
            for label, count in label_counts.most_common()
        ])
        
        # Add cumulative percentage
        distribution_df['cumulative_percentage'] = distribution_df['percentage'].cumsum()
        
        # Save complete distribution to CSV
        distribution_df.to_csv(self.output_dir / f'{self.framework}_complete_label_distribution.csv', index=False)
        
        # Calculate distribution metrics
        total_labels = len(label_counts)
        label_values = list(label_counts.values())
        
        # Gini coefficient
        sorted_values = sorted(label_values)
        n = len(sorted_values)
        index = np.arange(1, n + 1)
        gini = (2 * index @ sorted_values) / (n * sum(sorted_values)) - (n + 1) / n
        
        # Entropy
        probs = np.array(label_values) / sum(label_values)
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        
        # Find how many labels for different coverage levels
        coverage_analysis = {}
        for target in [50, 80, 90, 95, 99]:
            cumsum = 0
            for i, (label, count) in enumerate(label_counts.most_common()):
                cumsum += count
                if cumsum >= sum(label_values) * target / 100:
                    coverage_analysis[f'labels_for_{target}pct'] = i + 1
                    break
        
        # Power law analysis
        try:
            from scipy import stats
            # Fit power law to label frequencies
            x = np.arange(1, len(label_values) + 1)
            y = sorted(label_values, reverse=True)
            log_x = np.log(x)
            log_y = np.log(y)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
            power_law_exponent = -slope
            power_law_r2 = r_value ** 2
        except:
            power_law_exponent = None
            power_law_r2 = None
        
        # Store results
        self.results['complete_distribution'] = {
            'total_unique_labels': total_labels,
            'total_label_instances': sum(label_values),
            'gini_coefficient': gini,
            'entropy': entropy,
            'max_label_count': max(label_values),
            'min_label_count': min(label_values),
            'mean_label_count': np.mean(label_values),
            'median_label_count': np.median(label_values),
            'power_law_exponent': power_law_exponent,
            'power_law_r2': power_law_r2,
            **coverage_analysis
        }
        
        # Print summary
        print(f"\n{self.framework} - Complete Label Distribution Summary:")
        print(f"Total unique labels: {total_labels}")
        print(f"Gini coefficient: {gini:.3f}")
        print(f"Entropy: {entropy:.3f}")
        print(f"Labels for 50% coverage: {coverage_analysis.get('labels_for_50pct', 'N/A')}")
        print(f"Labels for 80% coverage: {coverage_analysis.get('labels_for_80pct', 'N/A')}")
        print(f"Labels for 95% coverage: {coverage_analysis.get('labels_for_95pct', 'N/A')}")
        
        # Print top 20 labels with counts
        print(f"\nTop 20 labels:")
        for i, (label, count) in enumerate(label_counts.most_common(20)):
            print(f"{i+1:2d}. {label:30s} - {count:5d} ({count/len(self.df)*100:5.2f}%)")
        
        # Visualize
        self._visualize_complete_distribution(label_counts, distribution_df, coverage_analysis)
        
    def _visualize_complete_distribution(self, label_counts, distribution_df, coverage_analysis):
        """Visualize complete label distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.framework} - Complete Label Distribution Analysis', fontsize=16, y=1.02)
        
        # 1. Distribution curve (log-log plot)
        ax = axes[0, 0]
        values = sorted(label_counts.values(), reverse=True)
        x = np.arange(1, len(values) + 1)
        ax.loglog(x, values, 'b-', alpha=0.7, linewidth=2)
        ax.set_xlabel('Label Rank')
        ax.set_ylabel('Usage Count')
        ax.set_title('Label Usage Distribution (Log-Log)')
        ax.grid(True, alpha=0.3, which='both')
        
        # Add power law fit if available
        if self.results['complete_distribution']['power_law_exponent']:
            exp = self.results['complete_distribution']['power_law_exponent']
            ax.text(0.1, 0.9, f'Power law exponent: {exp:.2f}', 
                   transform=ax.transAxes, fontsize=10)
        
        # 2. Cumulative distribution
        ax = axes[0, 1]
        ax.plot(range(len(distribution_df)), distribution_df['cumulative_percentage'], 
               'g-', linewidth=2)
        
        # Mark coverage points
        for pct, n_labels in coverage_analysis.items():
            target = int(pct.split('_')[2].replace('pct', ''))
            ax.axhline(target, color='red', linestyle='--', alpha=0.5)
            ax.axvline(n_labels, color='red', linestyle='--', alpha=0.5)
            ax.text(n_labels + 5, target - 2, f'{target}%', fontsize=8)
        
        ax.set_xlabel('Number of Labels')
        ax.set_ylabel('Cumulative Coverage (%)')
        ax.set_title('Cumulative Label Coverage')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(100, len(distribution_df)))
        
        # 3. Label frequency histogram
        ax = axes[1, 0]
        # Group by frequency ranges
        freq_bins = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10000]
        freq_counts = pd.cut(distribution_df['count'], bins=freq_bins, right=False).value_counts().sort_index()
        
        ax.bar(range(len(freq_counts)), freq_counts.values, color='orange', alpha=0.7)
        ax.set_xticks(range(len(freq_counts)))
        ax.set_xticklabels([str(interval) for interval in freq_counts.index], rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Usage Count Range')
        ax.set_ylabel('Number of Labels')
        ax.set_title('Label Frequency Distribution')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 4. Distribution metrics
        ax = axes[1, 1]
        metrics = self.results['complete_distribution']
        
        # Create a table of metrics
        metric_data = [
            ['Total Labels', f"{metrics['total_unique_labels']:,}"],
            ['Gini Coefficient', f"{metrics['gini_coefficient']:.3f}"],
            ['Entropy', f"{metrics['entropy']:.3f}"],
            ['Labels for 50%', f"{metrics.get('labels_for_50pct', 'N/A')}"],
            ['Labels for 80%', f"{metrics.get('labels_for_80pct', 'N/A')}"],
            ['Labels for 95%', f"{metrics.get('labels_for_95pct', 'N/A')}"],
            ['Mean Usage', f"{metrics['mean_label_count']:.1f}"],
            ['Median Usage', f"{metrics['median_label_count']:.0f}"]
        ]
        
        # Hide axes
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=metric_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(metric_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('Distribution Metrics Summary')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_complete_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_label_health(self):
        """Section 14: Label System Health Metrics"""
        print(f"[{self.framework}] Analyzing label system health...")
        
        # Calculate entropy over time
        monthly_entropy = []
        monthly_label_counts = []
        
        for month in pd.date_range(self.df['created_at'].min(), self.df['created_at'].max(), freq='M'):
            month_start = month.replace(day=1)
            month_end = (month + pd.DateOffset(months=1)).replace(day=1)
            
            month_df = self.df[(self.df['created_at'] >= month_start) & 
                               (self.df['created_at'] < month_end)]
            
            if len(month_df) > 10:  # Minimum threshold
                month_labels = []
                for labels in month_df['label_names']:
                    month_labels.extend(labels)
                
                if month_labels:
                    label_counts = Counter(month_labels)
                    probs = np.array(list(label_counts.values())) / sum(label_counts.values())
                    entropy = -sum(p * np.log(p) for p in probs if p > 0)
                    
                    monthly_entropy.append({
                        'month': month,
                        'entropy': entropy,
                        'unique_labels': len(label_counts),
                        'total_issues': len(month_df)
                    })
                    monthly_label_counts.append(label_counts)
        
        # Label lifecycle analysis
        label_lifecycles = self._analyze_label_lifecycles(monthly_label_counts)
        
        # Cross-team usage analysis
        team_usage = self._analyze_team_usage()
        
        # Store results
        if monthly_entropy:
            entropy_trend = np.polyfit(range(len(monthly_entropy)), 
                                     [m['entropy'] for m in monthly_entropy], 1)[0]
        else:
            entropy_trend = 0
            
        self.results['health'] = {
            'entropy_trend': entropy_trend,
            'current_entropy': monthly_entropy[-1]['entropy'] if monthly_entropy else None,
            'lifecycle_distribution': label_lifecycles['distribution'],
            'dying_labels': len(label_lifecycles['dying']),
            'growing_labels': len(label_lifecycles['growing']),
            'team_consistency': team_usage['consistency_score']
        }
        
        # Visualize
        self._visualize_health(monthly_entropy, label_lifecycles, team_usage)
        
    def _analyze_label_lifecycles(self, monthly_label_counts):
        """Classify labels by lifecycle stage"""
        if len(monthly_label_counts) < 6:
            return {'distribution': {}, 'dying': [], 'growing': []}
            
        # Track label usage over time
        label_timeseries = defaultdict(list)
        
        for month_counts in monthly_label_counts:
            all_labels = set()
            for counts in monthly_label_counts:
                all_labels.update(counts.keys())
            
            for label in all_labels:
                label_timeseries[label].append(month_counts.get(label, 0))
        
        # Classify each label
        lifecycles = {'new': [], 'growing': [], 'stable': [], 'declining': [], 'dead': [], 'dying': []}
        
        for label, usage in label_timeseries.items():
            if sum(usage) < 10:  # Too few to classify
                continue
                
            # Recent usage (last 3 months)
            recent = usage[-3:] if len(usage) >= 3 else usage
            # Historical usage (3-6 months ago)
            historical = usage[-6:-3] if len(usage) >= 6 else usage[:len(usage)-3]
            
            recent_avg = np.mean(recent) if recent else 0
            historical_avg = np.mean(historical) if historical else 0
            
            # Classification logic
            if sum(usage[:-3]) == 0 and recent_avg > 0:
                lifecycles['new'].append(label)
            elif recent_avg == 0 and historical_avg > 0:
                lifecycles['dead'].append(label)
            elif recent_avg < historical_avg * 0.3:
                lifecycles['dying'].append(label)
            elif recent_avg > historical_avg * 1.5:
                lifecycles['growing'].append(label)
            elif recent_avg < historical_avg * 0.7:
                lifecycles['declining'].append(label)
            else:
                lifecycles['stable'].append(label)
        
        return {
            'distribution': {k: len(v) for k, v in lifecycles.items()},
            'dying': lifecycles['dying'][:10],
            'growing': lifecycles['growing'][:10],
            'lifecycles': lifecycles
        }
        
    def _analyze_team_usage(self):
        """Analyze how different teams use labels"""
        team_label_usage = defaultdict(lambda: defaultdict(int))
        
        for _, row in self.df.iterrows():
            author_type = row.get('author_association', 'NONE')
            for label in row['label_names']:
                team_label_usage[author_type][label] += 1
        
        # Calculate consistency scores
        all_labels = set()
        for labels in team_label_usage.values():
            all_labels.update(labels.keys())
        
        # Create usage vectors for each team
        team_vectors = {}
        for team, usage in team_label_usage.items():
            vector = [usage.get(label, 0) for label in sorted(all_labels)]
            if sum(vector) > 0:
                # Normalize
                vector = np.array(vector) / sum(vector)
                team_vectors[team] = vector
        
        # Calculate pairwise similarities
        similarities = []
        teams = list(team_vectors.keys())
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                # Cosine similarity
                sim = np.dot(team_vectors[teams[i]], team_vectors[teams[j]])
                similarities.append(sim)
        
        consistency_score = np.mean(similarities) if similarities else 0
        
        return {
            'consistency_score': consistency_score,
            'team_count': len(team_vectors),
            'team_usage': dict(team_label_usage)
        }
        
    def _visualize_health(self, monthly_entropy, label_lifecycles, team_usage):
        """Visualize label system health metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.framework} - Label System Health Analysis', fontsize=16, y=1.02)
        
        # 1. Entropy over time
        ax = axes[0, 0]
        if monthly_entropy:
            months = [m['month'] for m in monthly_entropy]
            entropies = [m['entropy'] for m in monthly_entropy]
            unique_labels = [m['unique_labels'] for m in monthly_entropy]
            
            ax.plot(months, entropies, 'b-', linewidth=2, label='Entropy')
            
            # Add trend line
            x = np.arange(len(entropies))
            z = np.polyfit(x, entropies, 1)
            p = np.poly1d(z)
            ax.plot(months, p(x), 'r--', alpha=0.7, label=f'Trend ({z[0]:.3f})')
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Entropy')
            ax.set_title('Label System Entropy Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add unique label count on secondary axis
            ax2 = ax.twinx()
            ax2.plot(months, unique_labels, 'g-', alpha=0.5, label='Unique Labels')
            ax2.set_ylabel('Unique Labels', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
        
        # 2. Label lifecycle distribution
        ax = axes[0, 1]
        if label_lifecycles['distribution']:
            stages = list(label_lifecycles['distribution'].keys())
            counts = list(label_lifecycles['distribution'].values())
            colors = ['#2ecc71', '#3498db', '#95a5a6', '#f39c12', '#e74c3c', '#c0392b']
            
            wedges, texts, autotexts = ax.pie(counts, labels=stages, colors=colors[:len(stages)], 
                                              autopct='%1.0f%%', startangle=90)
            ax.set_title('Label Lifecycle Stage Distribution')
        
        # 3. Growing vs Dying labels
        ax = axes[1, 0]
        growing = label_lifecycles.get('growing', [])[:10]
        dying = label_lifecycles.get('dying', [])[:10]
        
        if growing or dying:
            # Create side-by-side bars
            y_pos = np.arange(max(len(growing), len(dying)))
            
            # Plot growing labels (positive side)
            for i, label in enumerate(growing):
                ax.barh(i, 1, color='green', alpha=0.7)
                ax.text(0.5, i, label, ha='center', va='center', fontsize=8)
            
            # Plot dying labels (negative side)
            for i, label in enumerate(dying):
                ax.barh(i, -1, color='red', alpha=0.7)
                ax.text(-0.5, i, label, ha='center', va='center', fontsize=8)
            
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.5, max(len(growing), len(dying)) - 0.5)
            ax.axvline(0, color='black', linewidth=1)
            ax.set_xlabel('← Dying | Growing →')
            ax.set_title('Growing vs Dying Labels')
            ax.set_yticks([])
        
        # 4. Team usage consistency
        ax = axes[1, 1]
        team_usage_data = team_usage.get('team_usage', {})
        if team_usage_data:
            # Show label usage by team type
            teams = list(team_usage_data.keys())[:5]  # Top 5 team types
            
            # Get top 10 labels across all teams
            all_team_labels = Counter()
            for team_labels in team_usage_data.values():
                all_team_labels.update(team_labels)
            top_labels = [label for label, _ in all_team_labels.most_common(10)]
            
            # Create heatmap data
            heatmap_data = []
            for team in teams:
                row = []
                team_total = sum(team_usage_data[team].values())
                for label in top_labels:
                    count = team_usage_data[team].get(label, 0)
                    row.append(count / team_total * 100 if team_total > 0 else 0)
                heatmap_data.append(row)
            
            if heatmap_data:
                sns.heatmap(heatmap_data, xticklabels=top_labels, yticklabels=teams,
                           cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Usage %'})
                ax.set_title(f'Label Usage by Team (Consistency: {team_usage["consistency_score"]:.2f})')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_label_health.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_report(self):
        """Generate comprehensive analysis report"""
        report = f"""# {self.framework} - Advanced Label Analysis Report

## 1. Label Anomalies
- Zombie labels (unused in last 6 months): {self.results['anomalies']['zombie_labels']}
- Orphaned labels (only on closed issues): {self.results['anomalies']['orphaned_labels']}
- Solo labels (never co-occur): {self.results['anomalies']['solo_labels']}
- Significant usage spikes detected: {self.results['anomalies']['label_spikes']}

## 2. Label Quality Metrics
- Average labeling delay: {self.results['quality'].get('avg_labeling_delay_hours', 'N/A'):.1f} hours
- Issues without labels: {self.results['quality']['no_label_ratio']*100:.1f}%
- Single label issues: {self.results['quality']['single_label_ratio']*100:.1f}%
- Multi-label issues: {self.results['quality']['multi_label_ratio']*100:.1f}%
- Average labels per issue: {self.results['quality']['avg_labels_per_issue']:.2f}

## 3. Performance Bottlenecks
- High stall rate labels: {self.results['bottlenecks']['high_stall_labels']}
- Comment explosion labels: {self.results['bottlenecks']['comment_explosion_labels']}
- Problematic label combinations: {self.results['bottlenecks']['problematic_combinations']}
"""
        
        if self.results['bottlenecks']['worst_stall_label']:
            report += f"- Worst stall label: {self.results['bottlenecks']['worst_stall_label']}\n"
        if self.results['bottlenecks']['worst_comment_label']:
            report += f"- Worst comment explosion label: {self.results['bottlenecks']['worst_comment_label']}\n"
            
        report += f"""
## 4. Label Optimization Opportunities
- Redundant label pairs identified: {self.results['optimization']['redundant_pairs']}
- Labels needed for 50% coverage: {self.results['optimization']['minimal_set_50']}
- Labels needed for 80% coverage: {self.results['optimization']['minimal_set_80']}
- Labels needed for 95% coverage: {self.results['optimization']['minimal_set_95']}
- Total labels in system: {self.results['optimization']['total_labels']}

## 5. Complete Distribution Statistics
- Total unique labels: {self.results['complete_distribution']['total_unique_labels']}
- Gini coefficient: {self.results['complete_distribution']['gini_coefficient']:.3f}
- Entropy: {self.results['complete_distribution']['entropy']:.3f}
- Power law exponent: {self.results['complete_distribution'].get('power_law_exponent', 'N/A')}
- Mean label usage: {self.results['complete_distribution']['mean_label_count']:.1f}
- Median label usage: {self.results['complete_distribution']['median_label_count']:.0f}

## 6. Label System Health
- Entropy trend: {self.results['health']['entropy_trend']:.3f} {'(increasing complexity)' if self.results['health']['entropy_trend'] > 0 else '(decreasing complexity)'}
- Current entropy: {self.results['health'].get('current_entropy', 'N/A')}
- Dying labels: {self.results['health']['dying_labels']}
- Growing labels: {self.results['health']['growing_labels']}
- Team consistency score: {self.results['health']['team_consistency']:.2f}

## Key Recommendations

1. **Label Cleanup**: Remove {self.results['anomalies']['zombie_labels']} zombie labels that haven't been used recently
2. **Redundancy Reduction**: Consider merging {self.results['optimization']['redundant_pairs']} redundant label pairs
3. **Process Improvement**: Focus on {self.results['bottlenecks']['high_stall_labels']} labels with high stall rates
4. **Simplification**: Current system uses {self.results['optimization']['total_labels']} labels but only {self.results['optimization']['minimal_set_80']} are needed for 80% coverage

## Files Generated
- Complete label distribution: {self.framework}_complete_label_distribution.csv
- Visualizations: *_label_*.png files
"""
        
        return report
        
    def run(self):
        """Run all advanced label analyses"""
        self.load_data()
        
        print("\nRunning advanced label analyses...")
        self.analyze_label_anomalies()
        self.analyze_label_quality()
        self.analyze_performance_bottlenecks()
        self.analyze_label_optimization()
        self.analyze_complete_distribution()
        self.analyze_label_health()
        
        # Generate and save report
        report = self.generate_report()
        with open(self.output_dir / f'{self.framework}_advanced_label_report.md', 'w') as f:
            f.write(report)
        
        # Save detailed results
        with open(self.output_dir / f'{self.framework}_advanced_label_data.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nAnalysis complete for {self.framework}")
        print(f"Report saved to: {self.output_dir / f'{self.framework}_advanced_label_report.md'}")
        print(f"Complete distribution saved to: {self.output_dir / f'{self.framework}_complete_label_distribution.csv'}")
        
        return self.results


if __name__ == "__main__":
    # Run analysis for all frameworks
    frameworks = [
        ('vllm', '/root/yunwei37/vllm-exp/bug-study/data/vllm_all_issues.json'),
        ('sglang', '/root/yunwei37/vllm-exp/bug-study/data/sglang_issues.json'),
        ('llama_cpp', '/root/yunwei37/vllm-exp/bug-study/data/llama_cpp_issues.json')
    ]
    
    output_base = Path('/root/yunwei37/vllm-exp/bug-study/analysis_basic/results/advanced_labels')
    
    for framework, data_path in frameworks:
        print(f"\n{'='*60}")
        print(f"Analyzing {framework}")
        print('='*60)
        
        analyzer = AdvancedLabelAnalyzer(framework, data_path, output_base)
        analyzer.run()