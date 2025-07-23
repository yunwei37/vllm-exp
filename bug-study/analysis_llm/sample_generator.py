#!/usr/bin/env python3
"""
Sample Generator for LLM Analysis
Generates various sampling groups from issue data for LLM-based analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')


class IssueSampler:
    def __init__(self, framework_name, data_path, output_base_dir):
        self.framework = framework_name
        self.data_path = Path(data_path)
        self.output_base = Path(output_base_dir) / framework_name
        self.df = None
        self.samples = {}
        
    def load_data(self):
        """Load and preprocess issue data"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        items = data.get('items', data if isinstance(data, list) else [])
        self.df = pd.DataFrame(items)
        
        # Convert timestamps
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
        self.df['closed_at'] = pd.to_datetime(self.df['closed_at'], errors='coerce')
        self.df['updated_at'] = pd.to_datetime(self.df['updated_at'], errors='coerce')
        
        # Extract label names
        self.df['label_names'] = self.df['labels'].apply(
            lambda x: [l['name'] for l in x if isinstance(l, dict) and 'name' in l] if isinstance(x, list) else []
        )
        
        # Calculate resolution time
        self.df['is_closed'] = self.df['state'] == 'closed'
        mask = self.df['is_closed'] & self.df['closed_at'].notna()
        self.df['resolution_hours'] = pd.NaT
        self.df.loc[mask, 'resolution_hours'] = (
            self.df.loc[mask, 'closed_at'] - self.df.loc[mask, 'created_at']
        ).dt.total_seconds() / 3600
        
        # Extract total reactions
        self.df['total_reactions'] = self.df['reactions'].apply(
            lambda x: sum(v for k, v in x.items() if k != 'url' and isinstance(v, int)) if isinstance(x, dict) else 0
        )
        
        # Calculate issue age in days
        current_time = pd.Timestamp.now(tz='UTC')
        # Make created_at timezone-aware if it isn't already
        if self.df['created_at'].dt.tz is None:
            self.df['created_at'] = self.df['created_at'].dt.tz_localize('UTC')
        self.df['age_days'] = (current_time - self.df['created_at']).dt.total_seconds() / 86400
        
        # Extract body length
        self.df['body_length'] = self.df['body'].fillna('').str.len()
        
        print(f"Loaded {len(self.df)} issues for {self.framework}")
        
    def save_sample(self, sample_name, issues_df, method_type):
        """Save a sample to structured directory"""
        # Create directory structure
        output_dir = self.output_base / method_type / sample_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sample data
        sample_data = []
        for _, issue in issues_df.iterrows():
            sample_data.append({
                'number': issue['number'],
                'title': issue['title'],
                'body': issue['body'],
                'labels': issue['label_names'],
                'state': issue['state'],
                'created_at': issue['created_at'].isoformat() if pd.notna(issue['created_at']) else None,
                'closed_at': issue['closed_at'].isoformat() if pd.notna(issue['closed_at']) else None,
                'comments': issue['comments'],
                'reactions': issue['reactions'],
                'author_association': issue.get('author_association', 'NONE'),
                'html_url': issue['html_url']
            })
        
        # Save as JSON
        with open(output_dir / 'issues.json', 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        # Save metadata
        metadata = {
            'framework': self.framework,
            'method_type': method_type,
            'sample_name': sample_name,
            'sample_size': len(sample_data),
            'generated_at': datetime.now().isoformat(),
            'sampling_criteria': self._get_sampling_criteria(method_type, sample_name)
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save issue list for quick reference
        with open(output_dir / 'issue_numbers.txt', 'w') as f:
            for issue in sample_data:
                f.write(f"{issue['number']}\n")
        
        print(f"  Saved {len(sample_data)} issues to {method_type}/{sample_name}")
        
    def _get_sampling_criteria(self, method_type, sample_name):
        """Get human-readable sampling criteria"""
        criteria_map = {
            'label_based': f"Issues with label '{sample_name}'",
            'temporal': f"Issues from {sample_name} time period",
            'resolution_time': f"Issues with {sample_name} resolution time",
            'complexity': f"Issues with {sample_name} complexity level",
            'author': f"Issues from {sample_name} authors",
            'reaction': f"Issues with {sample_name} reaction level",
            'long_tail': f"Issues from {sample_name} label frequency group",
            'cross_reference': f"Issues with {sample_name} references",
            'state_transition': f"Issues with {sample_name} state transitions",
            'anomaly': f"Issues with {sample_name} anomaly pattern"
        }
        return criteria_map.get(method_type, sample_name)
        
    def sample_by_labels(self):
        """Method 1: Label-based sampling"""
        print("\n1. Label-based sampling...")
        
        # Get label counts
        all_labels = []
        for labels in self.df['label_names']:
            all_labels.extend(labels)
        label_counts = Counter(all_labels)
        
        for label, count in label_counts.most_common():
            # Get issues with this label
            label_issues = self.df[self.df['label_names'].apply(lambda x: label in x)]
            
            # Sample appropriately
            if count >= 30:
                sampled = label_issues.sample(n=30, random_state=42)
            else:
                sampled = label_issues
            
            self.save_sample(label.replace('/', '_').replace(' ', '_'), sampled, 'label_based')
            
    def sample_by_temporal(self):
        """Method 2: Temporal-based sampling"""
        print("\n2. Temporal-based sampling...")
        
        now = pd.Timestamp.now(tz='UTC')
        
        # Recent burst - last 30 days
        recent_mask = self.df['created_at'] >= (now - timedelta(days=30))
        recent_issues = self.df[recent_mask]
        if len(recent_issues) >= 30:
            self.save_sample('recent_burst_30days', recent_issues.sample(n=30, random_state=42), 'temporal')
        else:
            self.save_sample('recent_burst_30days', recent_issues, 'temporal')
        
        # Steady state - 6 months ago
        six_months_ago = now - timedelta(days=180)
        steady_mask = (self.df['created_at'] >= (six_months_ago - timedelta(days=30))) & \
                     (self.df['created_at'] < six_months_ago)
        steady_issues = self.df[steady_mask]
        if len(steady_issues) >= 30:
            self.save_sample('steady_state_6months', steady_issues.sample(n=30, random_state=42), 'temporal')
        else:
            self.save_sample('steady_state_6months', steady_issues, 'temporal')
        
        # Historical - 1+ year ago
        one_year_ago = now - timedelta(days=365)
        historical_mask = self.df['created_at'] < one_year_ago
        historical_issues = self.df[historical_mask]
        if len(historical_issues) >= 30:
            self.save_sample('historical_1year_plus', historical_issues.sample(n=30, random_state=42), 'temporal')
        else:
            self.save_sample('historical_1year_plus', historical_issues, 'temporal')
        
        # First month of project
        first_date = self.df['created_at'].min()
        first_month_mask = self.df['created_at'] < (first_date + timedelta(days=30))
        first_month_issues = self.df[first_month_mask]
        self.save_sample('first_month', first_month_issues, 'temporal')
        
    def sample_by_resolution_time(self):
        """Method 3: Resolution time-based sampling"""
        print("\n3. Resolution time-based sampling...")
        
        closed_issues = self.df[self.df['is_closed'] & self.df['resolution_hours'].notna()]
        
        # Define resolution time buckets
        buckets = [
            ('lightning_fast_1hour', 0, 1),
            ('quick_1to24hours', 1, 24),
            ('normal_1to7days', 24, 24*7),
            ('slow_7to30days', 24*7, 24*30),
            ('very_slow_over30days', 24*30, float('inf'))
        ]
        
        for bucket_name, min_hours, max_hours in buckets:
            if max_hours == float('inf'):
                bucket_issues = closed_issues[closed_issues['resolution_hours'] > min_hours]
            else:
                bucket_issues = closed_issues[
                    (closed_issues['resolution_hours'] > min_hours) & 
                    (closed_issues['resolution_hours'] <= max_hours)
                ]
            
            if len(bucket_issues) >= 30:
                self.save_sample(bucket_name, bucket_issues.sample(n=30, random_state=42), 'resolution_time')
            else:
                self.save_sample(bucket_name, bucket_issues, 'resolution_time')
        
        # Never closed - open > 180 days
        never_closed_mask = (~self.df['is_closed']) & (self.df['age_days'] > 180)
        never_closed_issues = self.df[never_closed_mask]
        if len(never_closed_issues) >= 30:
            self.save_sample('never_closed_180days', never_closed_issues.sample(n=30, random_state=42), 'resolution_time')
        else:
            self.save_sample('never_closed_180days', never_closed_issues, 'resolution_time')
            
    def sample_by_complexity(self):
        """Method 4: Complexity-based sampling (using comments as proxy)"""
        print("\n4. Complexity-based sampling...")
        
        buckets = [
            ('zero_comments', 0, 0),
            ('low_discussion_1to5', 1, 5),
            ('medium_discussion_6to20', 6, 20),
            ('high_discussion_21to50', 21, 50),
            ('very_high_discussion_over50', 51, float('inf'))
        ]
        
        for bucket_name, min_comments, max_comments in buckets:
            if min_comments == 0 and max_comments == 0:
                bucket_issues = self.df[self.df['comments'] == 0]
            elif max_comments == float('inf'):
                bucket_issues = self.df[self.df['comments'] >= min_comments]
            else:
                bucket_issues = self.df[
                    (self.df['comments'] >= min_comments) & 
                    (self.df['comments'] <= max_comments)
                ]
            
            if len(bucket_issues) >= 30:
                self.save_sample(bucket_name, bucket_issues.sample(n=30, random_state=42), 'complexity')
            else:
                self.save_sample(bucket_name, bucket_issues, 'complexity')
                
    def sample_by_author(self):
        """Method 5: Author-based sampling"""
        print("\n5. Author-based sampling...")
        
        # Count issues per author
        author_counts = self.df['user'].apply(lambda x: x['login'] if isinstance(x, dict) else 'unknown').value_counts()
        
        # First-time contributors (authors with only 1 issue)
        first_timers = author_counts[author_counts == 1].index
        first_timer_issues = self.df[self.df['user'].apply(
            lambda x: x['login'] if isinstance(x, dict) else 'unknown').isin(first_timers)
        ]
        if len(first_timer_issues) >= 30:
            self.save_sample('first_time_contributors', first_timer_issues.sample(n=30, random_state=42), 'author')
        else:
            self.save_sample('first_time_contributors', first_timer_issues, 'author')
        
        # Regular contributors (5-20 issues)
        regular_authors = author_counts[(author_counts >= 5) & (author_counts <= 20)].index
        regular_issues = self.df[self.df['user'].apply(
            lambda x: x['login'] if isinstance(x, dict) else 'unknown').isin(regular_authors)
        ]
        if len(regular_issues) >= 30:
            self.save_sample('regular_contributors_5to20', regular_issues.sample(n=30, random_state=42), 'author')
        else:
            self.save_sample('regular_contributors_5to20', regular_issues, 'author')
        
        # Power users (>20 issues)
        power_authors = author_counts[author_counts > 20].index
        power_issues = self.df[self.df['user'].apply(
            lambda x: x['login'] if isinstance(x, dict) else 'unknown').isin(power_authors)
        ]
        if len(power_issues) >= 30:
            self.save_sample('power_users_over20', power_issues.sample(n=30, random_state=42), 'author')
        else:
            self.save_sample('power_users_over20', power_issues, 'author')
        
        # By author association
        associations = ['MEMBER', 'CONTRIBUTOR', 'NONE']
        for assoc in associations:
            assoc_issues = self.df[self.df.get('author_association', 'NONE') == assoc]
            if len(assoc_issues) >= 30:
                self.save_sample(f'author_association_{assoc}', assoc_issues.sample(n=30, random_state=42), 'author')
            else:
                self.save_sample(f'author_association_{assoc}', assoc_issues, 'author')
                
    def sample_by_reactions(self):
        """Method 6: Reaction-based sampling"""
        print("\n6. Reaction-based sampling...")
        
        buckets = [
            ('high_impact_over10', 11, float('inf')),
            ('moderate_impact_3to10', 3, 10),
            ('low_impact_1to2', 1, 2),
            ('no_engagement_0', 0, 0)
        ]
        
        for bucket_name, min_reactions, max_reactions in buckets:
            if min_reactions == 0 and max_reactions == 0:
                bucket_issues = self.df[self.df['total_reactions'] == 0]
            elif max_reactions == float('inf'):
                bucket_issues = self.df[self.df['total_reactions'] >= min_reactions]
            else:
                bucket_issues = self.df[
                    (self.df['total_reactions'] >= min_reactions) & 
                    (self.df['total_reactions'] <= max_reactions)
                ]
            
            if len(bucket_issues) >= 30:
                self.save_sample(bucket_name, bucket_issues.sample(n=30, random_state=42), 'reaction')
            else:
                self.save_sample(bucket_name, bucket_issues, 'reaction')
                
    def sample_long_tail(self):
        """Method 7: Long-tail label sampling"""
        print("\n7. Long-tail label sampling...")
        
        # Get label counts
        all_labels = []
        for labels in self.df['label_names']:
            all_labels.extend(labels)
        label_counts = Counter(all_labels)
        
        # Sort labels by frequency
        sorted_labels = [label for label, count in label_counts.most_common()]
        total_labels = len(sorted_labels)
        
        if total_labels == 0:
            return
            
        # Head: top 5 labels
        head_labels = sorted_labels[:5]
        head_issues = []
        for label in head_labels:
            label_issues = self.df[self.df['label_names'].apply(lambda x: label in x)]
            if len(label_issues) >= 10:
                head_issues.extend(label_issues.sample(n=10, random_state=42).to_dict('records'))
            else:
                head_issues.extend(label_issues.to_dict('records'))
        
        if head_issues:
            head_df = pd.DataFrame(head_issues).drop_duplicates(subset=['number'])
            self.save_sample('head_top5_labels', head_df, 'long_tail')
        
        # Body: middle 50%
        body_start = 5
        body_end = min(body_start + int(total_labels * 0.5), total_labels)
        body_labels = sorted_labels[body_start:body_end]
        body_issues = []
        for label in body_labels[:10]:  # Sample from first 10 body labels
            label_issues = self.df[self.df['label_names'].apply(lambda x: label in x)]
            if len(label_issues) >= 3:
                body_issues.extend(label_issues.sample(n=3, random_state=42).to_dict('records'))
            else:
                body_issues.extend(label_issues.to_dict('records'))
        
        if body_issues:
            body_df = pd.DataFrame(body_issues).drop_duplicates(subset=['number'])
            self.save_sample('body_middle50pct_labels', body_df, 'long_tail')
        
        # Tail: bottom 25%
        tail_start = int(total_labels * 0.75)
        tail_labels = sorted_labels[tail_start:]
        tail_issues = []
        for label in tail_labels:
            label_issues = self.df[self.df['label_names'].apply(lambda x: label in x)]
            tail_issues.extend(label_issues.to_dict('records'))
        
        if tail_issues:
            tail_df = pd.DataFrame(tail_issues).drop_duplicates(subset=['number'])
            self.save_sample('tail_bottom25pct_labels', tail_df, 'long_tail')
            
    def sample_cross_references(self):
        """Method 8: Cross-reference sampling"""
        print("\n8. Cross-reference sampling...")
        
        # Issues that reference PRs (look for #\d+ patterns in body)
        if 'body' in self.df.columns:
            import re
            pr_pattern = r'#\d+'
            has_refs = self.df['body'].fillna('').str.contains(pr_pattern, regex=True)
            ref_issues = self.df[has_refs]
            
            if len(ref_issues) >= 30:
                self.save_sample('references_prs', ref_issues.sample(n=30, random_state=42), 'cross_reference')
            else:
                self.save_sample('references_prs', ref_issues, 'cross_reference')
        
        # Duplicate issues (if duplicate label exists)
        dup_labels = ['duplicate', 'duplicated', 'dupe']
        dup_mask = self.df['label_names'].apply(
            lambda labels: any(dup in label.lower() for label in labels for dup in dup_labels)
        )
        dup_issues = self.df[dup_mask]
        if len(dup_issues) > 0:
            if len(dup_issues) >= 30:
                self.save_sample('duplicate_marked', dup_issues.sample(n=30, random_state=42), 'cross_reference')
            else:
                self.save_sample('duplicate_marked', dup_issues, 'cross_reference')
                
    def sample_state_transitions(self):
        """Method 9: State transition sampling"""
        print("\n9. State transition sampling...")
        
        # Stale to active (had 'stale' label)
        stale_labels = ['stale', 'inactive']
        stale_mask = self.df['label_names'].apply(
            lambda labels: any(stale in label.lower() for label in labels for stale in stale_labels)
        )
        stale_issues = self.df[stale_mask & ~self.df['is_closed']]  # Still open despite stale
        
        if len(stale_issues) >= 30:
            self.save_sample('stale_but_active', stale_issues.sample(n=30, random_state=42), 'state_transition')
        else:
            self.save_sample('stale_but_active', stale_issues, 'state_transition')
        
        # High label churn (many different labels)
        high_churn = self.df[self.df['label_names'].apply(len) > 3]
        if len(high_churn) >= 30:
            self.save_sample('high_label_churn_over3', high_churn.sample(n=30, random_state=42), 'state_transition')
        else:
            self.save_sample('high_label_churn_over3', high_churn, 'state_transition')
            
    def sample_anomalies(self):
        """Method 10: Metadata anomaly sampling"""
        print("\n10. Anomaly sampling...")
        
        # Very short issues
        short_issues = self.df[self.df['body_length'] < 50]
        if len(short_issues) >= 30:
            self.save_sample('very_short_under50chars', short_issues.sample(n=30, random_state=42), 'anomaly')
        else:
            self.save_sample('very_short_under50chars', short_issues, 'anomaly')
        
        # Very long issues
        long_issues = self.df[self.df['body_length'] > 2000]
        if len(long_issues) >= 30:
            self.save_sample('very_long_over2000chars', long_issues.sample(n=30, random_state=42), 'anomaly')
        else:
            self.save_sample('very_long_over2000chars', long_issues, 'anomaly')
        
        # Quick close (created and closed within 1 hour)
        quick_close_mask = (self.df['is_closed']) & (self.df['resolution_hours'] < 1)
        quick_close_issues = self.df[quick_close_mask]
        if len(quick_close_issues) >= 30:
            self.save_sample('quick_close_under1hour', quick_close_issues.sample(n=30, random_state=42), 'anomaly')
        else:
            self.save_sample('quick_close_under1hour', quick_close_issues, 'anomaly')
            
    def generate_summary_report(self):
        """Generate a summary of all samples created"""
        summary = {
            'framework': self.framework,
            'total_issues': len(self.df),
            'sampling_methods': {},
            'generated_at': datetime.now().isoformat()
        }
        
        # Walk through output directory and collect sample info
        for method_dir in self.output_base.iterdir():
            if method_dir.is_dir():
                method_name = method_dir.name
                summary['sampling_methods'][method_name] = {}
                
                for sample_dir in method_dir.iterdir():
                    if sample_dir.is_dir() and (sample_dir / 'metadata.json').exists():
                        with open(sample_dir / 'metadata.json', 'r') as f:
                            metadata = json.load(f)
                            summary['sampling_methods'][method_name][sample_dir.name] = {
                                'sample_size': metadata['sample_size'],
                                'criteria': metadata['sampling_criteria']
                            }
        
        # Save summary
        with open(self.output_base / 'sampling_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Generate README
        readme_content = f"""# Sampling Results for {self.framework}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total issues in dataset: {len(self.df)}

## Sampling Methods Applied

"""
        
        for method, samples in summary['sampling_methods'].items():
            readme_content += f"### {method.replace('_', ' ').title()}\n\n"
            for sample_name, info in samples.items():
                readme_content += f"- **{sample_name}**: {info['sample_size']} issues\n"
                readme_content += f"  - Criteria: {info['criteria']}\n"
            readme_content += "\n"
            
        with open(self.output_base / 'README.md', 'w') as f:
            f.write(readme_content)
            
        print(f"\nSummary report saved to {self.output_base}/sampling_summary.json")
        
    def run_all_sampling(self):
        """Run all sampling methods"""
        print(f"\nGenerating samples for {self.framework}...")
        
        self.sample_by_labels()
        self.sample_by_temporal()
        self.sample_by_resolution_time()
        self.sample_by_complexity()
        self.sample_by_author()
        self.sample_by_reactions()
        self.sample_long_tail()
        self.sample_cross_references()
        self.sample_state_transitions()
        self.sample_anomalies()
        
        self.generate_summary_report()
        print(f"\nAll sampling completed for {self.framework}!")


def main():
    """Run sampling for all frameworks"""
    frameworks = [
        ('vllm', '/root/yunwei37/vllm-exp/bug-study/data/vllm_all_issues.json'),
        ('sglang', '/root/yunwei37/vllm-exp/bug-study/data/sglang_issues.json'),
        ('llama_cpp', '/root/yunwei37/vllm-exp/bug-study/data/llama_cpp_issues.json')
    ]
    
    output_base = Path('/root/yunwei37/vllm-exp/bug-study/analysis_llm/sample_results')
    
    for framework, data_path in frameworks:
        print(f"\n{'='*60}")
        print(f"Processing {framework}")
        print('='*60)
        
        sampler = IssueSampler(framework, data_path, output_base)
        sampler.load_data()
        sampler.run_all_sampling()
        
    print("\n✅ All sampling completed!")
    print(f"Results saved to: {output_base}")


if __name__ == "__main__":
    main()