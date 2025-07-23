#!/usr/bin/env python3
"""
User Behavior Analysis Module: Analyze contributor patterns and community dynamics
Research Questions:
- User contribution patterns and inequality
- User expertise evolution over time
- Community growth and retention
- Collaboration networks
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class UserBehaviorAnalyzer:
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
        
        # Extract user information
        self.df['user_login'] = self.df['user'].apply(lambda x: x.get('login') if isinstance(x, dict) else None)
        self.df['user_id'] = self.df['user'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
        
        # Extract comment count and labels
        self.df['label_count'] = self.df['labels'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        self.df['is_closed'] = self.df['state'] == 'closed'
        
    def analyze_user_distribution(self):
        """Analyze user contribution patterns and inequality"""
        print(f"[{self.framework}] Analyzing user distribution...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.framework} - User Contribution Analysis', fontsize=16, y=1.02)
        
        # Get user contribution counts
        user_counts = self.df['user_login'].value_counts()
        
        # 1. User contribution distribution (log scale)
        ax = axes[0, 0]
        ax.hist(user_counts.values, bins=50, edgecolor='black', alpha=0.7)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title('User Contribution Distribution (Log Scale)')
        ax.set_xlabel('Number of Issues Created')
        ax.set_ylabel('Number of Users')
        ax.grid(True, alpha=0.3)
        
        # 2. Lorenz curve and Gini coefficient
        ax = axes[0, 1]
        sorted_contributions = np.sort(user_counts.values)
        cumsum = np.cumsum(sorted_contributions)
        cumsum_normalized = cumsum / cumsum[-1]
        user_percentiles = np.arange(1, len(sorted_contributions) + 1) / len(sorted_contributions)
        
        # Calculate Gini coefficient
        gini = 1 - 2 * np.trapz(cumsum_normalized, user_percentiles)
        
        ax.plot(user_percentiles, cumsum_normalized, linewidth=2, label=f'Lorenz Curve (Gini={gini:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality')
        ax.fill_between(user_percentiles, cumsum_normalized, user_percentiles, alpha=0.3)
        ax.set_title('User Contribution Inequality')
        ax.set_xlabel('Cumulative % of Users')
        ax.set_ylabel('Cumulative % of Contributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Top contributors
        ax = axes[0, 2]
        top_users = user_counts.head(20)
        y_pos = np.arange(len(top_users))
        ax.barh(y_pos, top_users.values, color='steelblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_users.index, fontsize=8)
        ax.set_title('Top 20 Contributors')
        ax.set_xlabel('Number of Issues')
        ax.grid(True, axis='x', alpha=0.3)
        
        # 4. Author association distribution
        ax = axes[1, 0]
        association_counts = self.df['author_association'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(association_counts)))
        wedges, texts, autotexts = ax.pie(association_counts.values, labels=association_counts.index, 
                                          autopct='%1.1f%%', colors=colors)
        ax.set_title('Issues by Author Association')
        
        # 5. User retention analysis
        ax = axes[1, 1]
        user_first_last = {}
        for user in user_counts.index[:100]:  # Top 100 users
            user_issues = self.df[self.df['user_login'] == user]['created_at']
            if len(user_issues) > 1:
                user_first_last[user] = {
                    'first': user_issues.min(),
                    'last': user_issues.max(),
                    'span_days': (user_issues.max() - user_issues.min()).days,
                    'count': len(user_issues)
                }
        
        if user_first_last:
            retention_df = pd.DataFrame(user_first_last).T
            ax.scatter(retention_df['count'], retention_df['span_days'], alpha=0.6)
            ax.set_xlabel('Total Issues Created')
            ax.set_ylabel('Active Days Span')
            ax.set_title('User Engagement Duration (Top 100 Users)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # 6. New vs returning users over time
        ax = axes[1, 2]
        monthly_users = defaultdict(lambda: {'new': 0, 'returning': 0})
        seen_users = set()
        
        for _, row in self.df.sort_values('created_at').iterrows():
            month = row['created_at'].to_period('M')
            user = row['user_login']
            if user in seen_users:
                monthly_users[month]['returning'] += 1
            else:
                monthly_users[month]['new'] += 1
                seen_users.add(user)
        
        months = sorted(monthly_users.keys())
        new_counts = [monthly_users[m]['new'] for m in months]
        returning_counts = [monthly_users[m]['returning'] for m in months]
        
        x = np.arange(len(months))
        width = 0.35
        ax.bar(x - width/2, new_counts, width, label='New Users', alpha=0.7)
        ax.bar(x + width/2, returning_counts, width, label='Returning Users', alpha=0.7)
        ax.set_title('New vs Returning Users per Month')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Users')
        ax.set_xticks(x[::max(1, len(x)//10)])
        ax.set_xticklabels([str(m) for m in months[::max(1, len(x)//10)]], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_user_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['distribution'] = {
            'total_users': len(user_counts),
            'gini_coefficient': gini,
            'top_20_percent_share': user_counts.head(int(len(user_counts) * 0.2)).sum() / user_counts.sum(),
            'single_issue_users': (user_counts == 1).sum(),
            'single_issue_ratio': (user_counts == 1).sum() / len(user_counts)
        }
        
    def analyze_user_behavior_patterns(self):
        """Analyze detailed user behavior patterns"""
        print(f"[{self.framework}] Analyzing user behavior patterns...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.framework} - User Behavior Patterns', fontsize=16, y=1.02)
        
        # Get user statistics
        user_stats = self.df.groupby('user_login').agg({
            'number': 'count',
            'is_closed': 'mean',
            'comments': 'mean',
            'label_count': 'mean',
            'created_at': ['min', 'max']
        })
        user_stats.columns = ['issue_count', 'closure_rate', 'avg_comments', 'avg_labels', 'first_issue', 'last_issue']
        user_stats['active_days'] = (user_stats['last_issue'] - user_stats['first_issue']).dt.days
        
        # 1. Issue count vs closure rate
        ax = axes[0, 0]
        # Filter users with at least 5 issues for meaningful closure rate
        experienced_users = user_stats[user_stats['issue_count'] >= 5]
        if len(experienced_users) > 0:
            scatter = ax.scatter(experienced_users['issue_count'], experienced_users['closure_rate'], 
                               c=experienced_users['avg_comments'], cmap='viridis', alpha=0.6)
            ax.set_xlabel('Number of Issues Created')
            ax.set_ylabel('Closure Rate')
            ax.set_title('User Experience vs Issue Success')
            ax.set_xscale('log')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Avg Comments')
            ax.grid(True, alpha=0.3)
        
        # 2. Author association performance
        ax = axes[0, 1]
        association_stats = self.df.groupby('author_association').agg({
            'is_closed': 'mean',
            'comments': 'mean',
            'number': 'count'
        }).sort_values('is_closed', ascending=False)
        
        x = np.arange(len(association_stats))
        ax.bar(x, association_stats['is_closed'], alpha=0.7, label='Closure Rate')
        ax2 = ax.twinx()
        ax2.plot(x, association_stats['comments'], 'ro-', label='Avg Comments', markersize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(association_stats.index, rotation=45, ha='right')
        ax.set_ylabel('Closure Rate')
        ax2.set_ylabel('Average Comments')
        ax.set_title('Performance by Author Association')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add sample sizes
        for i, (idx, row) in enumerate(association_stats.iterrows()):
            ax.text(i, 0.05, f'n={row["number"]}', ha='center', fontsize=8)
        
        # 3. User activity timeline
        ax = axes[1, 0]
        # Select top 20 most active users
        top_users = user_stats.nlargest(20, 'issue_count').index
        
        for i, user in enumerate(top_users[:10]):  # Limit to 10 for readability
            user_issues = self.df[self.df['user_login'] == user]['created_at']
            y = [i] * len(user_issues)
            ax.scatter(user_issues, y, alpha=0.6, s=20)
        
        ax.set_yticks(range(10))
        ax.set_yticklabels(top_users[:10])
        ax.set_xlabel('Date')
        ax.set_title('Top User Activity Timeline')
        ax.grid(True, axis='x', alpha=0.3)
        
        # 4. User clustering by behavior
        ax = axes[1, 1]
        if len(experienced_users) > 10:
            # Simple 2D clustering visualization
            x = experienced_users['issue_count']
            y = experienced_users['closure_rate']
            
            # Define user categories
            power_users = experienced_users[(experienced_users['issue_count'] > experienced_users['issue_count'].quantile(0.75)) & 
                                           (experienced_users['closure_rate'] > 0.5)]
            struggling_users = experienced_users[(experienced_users['issue_count'] > experienced_users['issue_count'].quantile(0.75)) & 
                                               (experienced_users['closure_rate'] <= 0.5)]
            quality_users = experienced_users[(experienced_users['issue_count'] <= experienced_users['issue_count'].quantile(0.75)) & 
                                             (experienced_users['closure_rate'] > 0.7)]
            
            ax.scatter(x, y, alpha=0.3, label='All Users')
            if len(power_users) > 0:
                ax.scatter(power_users['issue_count'], power_users['closure_rate'], 
                          color='green', s=100, label=f'Power Users ({len(power_users)})')
            if len(struggling_users) > 0:
                ax.scatter(struggling_users['issue_count'], struggling_users['closure_rate'], 
                          color='red', s=100, label=f'High Volume, Low Success ({len(struggling_users)})')
            if len(quality_users) > 0:
                ax.scatter(quality_users['issue_count'], quality_users['closure_rate'], 
                          color='blue', s=100, label=f'Quality Contributors ({len(quality_users)})')
            
            ax.set_xlabel('Number of Issues')
            ax.set_ylabel('Closure Rate')
            ax.set_title('User Behavior Clusters')
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.framework}_user_behavior.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.results['behavior'] = {
            'avg_issues_per_user': user_stats['issue_count'].mean(),
            'median_issues_per_user': user_stats['issue_count'].median(),
            'avg_user_closure_rate': user_stats['closure_rate'].mean(),
            'power_user_count': len(power_users) if 'power_users' in locals() else 0
        }
        
    def analyze_collaboration_network(self):
        """Analyze user collaboration patterns through issue interactions"""
        print(f"[{self.framework}] Analyzing collaboration network...")
        
        # Build collaboration graph based on users commenting on same issues
        G = nx.Graph()
        issue_users = defaultdict(set)
        
        # Group users by issues they've interacted with
        for _, row in self.df.iterrows():
            issue_id = row['number']
            creator = row['user_login']
            if creator:
                issue_users[issue_id].add(creator)
        
        # Create edges between users who worked on same issues
        edge_weights = defaultdict(int)
        for users in issue_users.values():
            users_list = list(users)
            for i in range(len(users_list)):
                for j in range(i + 1, len(users_list)):
                    edge = tuple(sorted([users_list[i], users_list[j]]))
                    edge_weights[edge] += 1
        
        # Add edges to graph (only significant collaborations)
        for (u, v), weight in edge_weights.items():
            if weight >= 2:  # At least 2 collaborations
                G.add_edge(u, v, weight=weight)
        
        if len(G.nodes()) > 10:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f'{self.framework} - User Collaboration Network', fontsize=16, y=1.02)
            
            # 1. Full network visualization (top nodes only)
            ax = axes[0, 0]
            # Get top nodes by degree
            node_degrees = dict(G.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:50]
            subG = G.subgraph([node for node, _ in top_nodes])
            
            pos = nx.spring_layout(subG, k=1, iterations=50)
            node_sizes = [node_degrees[node] * 20 for node in subG.nodes()]
            nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, alpha=0.6, ax=ax)
            nx.draw_networkx_edges(subG, pos, alpha=0.2, ax=ax)
            ax.set_title('Collaboration Network (Top 50 Users)')
            ax.axis('off')
            
            # 2. Degree distribution
            ax = axes[0, 1]
            degrees = [d for _, d in G.degree()]
            ax.hist(degrees, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Number of Collaborators')
            ax.set_ylabel('Number of Users')
            ax.set_title('Collaboration Degree Distribution')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # 3. Network metrics
            ax = axes[1, 0]
            metrics = {
                'Nodes': len(G.nodes()),
                'Edges': len(G.edges()),
                'Avg Degree': np.mean(degrees),
                'Density': nx.density(G),
                'Components': nx.number_connected_components(G)
            }
            
            y_pos = np.arange(len(metrics))
            ax.barh(y_pos, list(metrics.values()), color='lightcoral', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(list(metrics.keys()))
            ax.set_title('Network Statistics')
            ax.set_xlabel('Value')
            
            # Add values on bars
            for i, (metric, value) in enumerate(metrics.items()):
                ax.text(value, i, f' {value:.3f}' if value < 1 else f' {int(value)}', 
                       va='center', fontsize=10)
            
            # 4. Community detection
            ax = axes[1, 1]
            if len(G.nodes()) > 10:
                # Find communities
                communities = list(nx.community.greedy_modularity_communities(G))
                community_sizes = [len(c) for c in communities]
                community_sizes.sort(reverse=True)
                
                ax.bar(range(min(20, len(community_sizes))), community_sizes[:20], 
                       color='skyblue', alpha=0.7)
                ax.set_xlabel('Community Rank')
                ax.set_ylabel('Community Size')
                ax.set_title('Community Size Distribution (Top 20)')
                ax.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{self.framework}_collaboration_network.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Store network results
            self.results['network'] = {
                'total_nodes': len(G.nodes()),
                'total_edges': len(G.edges()),
                'avg_degree': np.mean(degrees),
                'max_degree': max(degrees),
                'network_density': nx.density(G),
                'num_components': nx.number_connected_components(G),
                'largest_component_size': len(max(nx.connected_components(G), key=len))
            }
    
    def analyze_user_evolution(self):
        """Analyze how user behavior evolves over time"""
        print(f"[{self.framework}] Analyzing user evolution...")
        
        # Get users with sufficient history
        user_counts = self.df['user_login'].value_counts()
        active_users = user_counts[user_counts >= 10].index
        
        if len(active_users) > 5:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{self.framework} - User Evolution Analysis', fontsize=16, y=1.02)
            
            evolution_data = []
            
            for user in active_users[:50]:  # Analyze top 50 active users
                user_issues = self.df[self.df['user_login'] == user].sort_values('created_at')
                if len(user_issues) >= 10:
                    # Split into thirds
                    n = len(user_issues)
                    early = user_issues.iloc[:n//3]
                    middle = user_issues.iloc[n//3:2*n//3]
                    late = user_issues.iloc[2*n//3:]
                    
                    evolution_data.append({
                        'user': user,
                        'early_closure': early['is_closed'].mean(),
                        'middle_closure': middle['is_closed'].mean(),
                        'late_closure': late['is_closed'].mean(),
                        'early_comments': early['comments'].mean(),
                        'late_comments': late['comments'].mean(),
                        'early_labels': early['label_count'].mean(),
                        'late_labels': late['label_count'].mean(),
                        'total_issues': n
                    })
            
            if evolution_data:
                evo_df = pd.DataFrame(evolution_data)
                
                # 1. Closure rate evolution
                ax = axes[0, 0]
                x = np.arange(3)
                early_mean = evo_df['early_closure'].mean()
                middle_mean = evo_df['middle_closure'].mean()
                late_mean = evo_df['late_closure'].mean()
                
                ax.plot(x, [early_mean, middle_mean, late_mean], 'o-', linewidth=2, markersize=10, label='Average')
                
                # Add individual user trajectories (sample)
                for _, user_data in evo_df.head(10).iterrows():
                    ax.plot(x, [user_data['early_closure'], user_data['middle_closure'], 
                               user_data['late_closure']], alpha=0.3, linewidth=1)
                
                ax.set_xticks(x)
                ax.set_xticklabels(['Early', 'Middle', 'Recent'])
                ax.set_ylabel('Closure Rate')
                ax.set_title('Issue Success Rate Evolution')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # 2. Comment engagement evolution
                ax = axes[0, 1]
                improvement = (evo_df['late_comments'] - evo_df['early_comments']) / (evo_df['early_comments'] + 1)
                ax.hist(improvement, bins=20, edgecolor='black', alpha=0.7)
                ax.axvline(0, color='red', linestyle='--', alpha=0.7)
                ax.set_xlabel('Comment Change Rate')
                ax.set_ylabel('Number of Users')
                ax.set_title('Change in Comment Engagement')
                ax.grid(True, axis='y', alpha=0.3)
                
                # 3. Learning curve
                ax = axes[1, 0]
                # Users who improved vs didn't
                improved = evo_df[evo_df['late_closure'] > evo_df['early_closure']]
                not_improved = evo_df[evo_df['late_closure'] <= evo_df['early_closure']]
                
                ax.scatter(improved['total_issues'], 
                          improved['late_closure'] - improved['early_closure'],
                          alpha=0.6, label=f'Improved ({len(improved)})', color='green')
                ax.scatter(not_improved['total_issues'], 
                          not_improved['late_closure'] - not_improved['early_closure'],
                          alpha=0.6, label=f'Not Improved ({len(not_improved)})', color='red')
                
                ax.set_xlabel('Total Issues Created')
                ax.set_ylabel('Closure Rate Change')
                ax.set_title('Learning Effect vs Experience')
                ax.axhline(0, color='black', linestyle='-', alpha=0.3)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 4. Evolution summary
                ax = axes[1, 1]
                evolution_metrics = {
                    'Users Analyzed': len(evo_df),
                    'Improved Success': (evo_df['late_closure'] > evo_df['early_closure']).sum(),
                    'Increased Engagement': (evo_df['late_comments'] > evo_df['early_comments']).sum(),
                    'Better Labeling': (evo_df['late_labels'] > evo_df['early_labels']).sum()
                }
                
                y_pos = np.arange(len(evolution_metrics))
                values = list(evolution_metrics.values())
                ax.barh(y_pos, values, color='lightblue', alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(list(evolution_metrics.keys()))
                ax.set_xlabel('Count')
                ax.set_title('User Evolution Summary')
                
                # Add percentages
                total = values[0]
                for i in range(1, len(values)):
                    ax.text(values[i] + 0.5, i, f'{values[i]/total*100:.1f}%', va='center')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f'{self.framework}_user_evolution.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Store evolution results
                self.results['evolution'] = {
                    'users_analyzed': len(evo_df),
                    'improvement_rate': (evo_df['late_closure'] > evo_df['early_closure']).mean(),
                    'avg_closure_change': (evo_df['late_closure'] - evo_df['early_closure']).mean(),
                    'engagement_change': (evo_df['late_comments'] - evo_df['early_comments']).mean()
                }
    
    def generate_report(self):
        """Generate user behavior analysis report"""
        report = f"""# {self.framework} - User Behavior Analysis Report

## User Distribution
- Total unique users: {self.results['distribution']['total_users']}
- Gini coefficient: {self.results['distribution']['gini_coefficient']:.3f}
- Top 20% users create: {self.results['distribution']['top_20_percent_share']*100:.1f}% of issues
- Single-issue users: {self.results['distribution']['single_issue_users']} ({self.results['distribution']['single_issue_ratio']*100:.1f}%)

## Behavior Patterns
- Average issues per user: {self.results['behavior']['avg_issues_per_user']:.2f}
- Median issues per user: {self.results['behavior']['median_issues_per_user']:.0f}
- Average user closure rate: {self.results['behavior']['avg_user_closure_rate']*100:.1f}%
- Power users identified: {self.results['behavior']['power_user_count']}
"""
        
        if 'network' in self.results:
            report += f"""
## Collaboration Network
- Network nodes (users): {self.results['network']['total_nodes']}
- Network edges (collaborations): {self.results['network']['total_edges']}
- Average collaborators per user: {self.results['network']['avg_degree']:.2f}
- Network density: {self.results['network']['network_density']:.4f}
- Connected components: {self.results['network']['num_components']}
"""
        
        if 'evolution' in self.results:
            report += f"""
## User Evolution
- Users analyzed: {self.results['evolution']['users_analyzed']}
- Users who improved: {self.results['evolution']['improvement_rate']*100:.1f}%
- Average closure rate change: {self.results['evolution']['avg_closure_change']*100:+.1f}%
- Average engagement change: {self.results['evolution']['engagement_change']:+.2f} comments
"""
        
        return report
    
    def run(self):
        """Run all user behavior analyses"""
        self.load_data()
        self.analyze_user_distribution()
        self.analyze_user_behavior_patterns()
        self.analyze_collaboration_network()
        self.analyze_user_evolution()
        
        # Save report
        report = self.generate_report()
        with open(self.output_dir / f'{self.framework}_user_behavior_report.md', 'w') as f:
            f.write(report)
        
        # Save data
        with open(self.output_dir / f'{self.framework}_user_behavior_data.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return self.results


if __name__ == "__main__":
    # Example usage
    analyzer = UserBehaviorAnalyzer(
        'vllm',
        '/root/yunwei37/vllm-exp/bug-study/data/vllm_all_issues.json',
        '/root/yunwei37/vllm-exp/bug-study/analysis/results/user_behavior'
    )
    analyzer.run()