#!/usr/bin/env python3
"""
GitHub Issues Scraper v2 with Cursor-Based Pagination Support

This version handles both page-based and cursor-based pagination
to work with large repositories that have many issues.
"""

import requests
import json
import time
import sys
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
import pickle
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()


class GitHubIssuesScraperV2:
    """Enhanced scraper for GitHub issues with cursor-based pagination support"""
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: Optional[str] = None):
        """Initialize the scraper"""
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'GitHub-Issues-Scraper/2.0'
        })
        
        if token:
            self.session.headers.update({'Authorization': f'token {token}'})
            print("✓ Using authenticated requests (5,000 requests/hour)")
        else:
            print("⚠ Using unauthenticated requests (60 requests/hour)")
    
    def parse_link_header(self, link_header: str) -> Dict[str, str]:
        """Parse GitHub's Link header for pagination"""
        links = {}
        if link_header:
            for link in link_header.split(','):
                match = re.match(r'<([^>]+)>;\s*rel="([^"]+)"', link.strip())
                if match:
                    url, rel = match.groups()
                    links[rel] = url
        return links
    
    def check_rate_limit(self) -> Tuple[int, int, int]:
        """Check current rate limit status"""
        try:
            response = self.session.get(f"{self.BASE_URL}/rate_limit")
            
            if response.status_code == 200:
                data = response.json()
                if 'rate' in data and 'core' in data['rate']:
                    core = data['rate']['core']
                    return core['limit'], core['remaining'], core['reset']
                elif 'core' in data:
                    core = data['core']
                    return core['limit'], core['remaining'], core['reset']
            
            # Fallback to headers
            return (
                int(response.headers.get('X-RateLimit-Limit', 5000 if self.session.headers.get('Authorization') else 60)),
                int(response.headers.get('X-RateLimit-Remaining', 50)),
                int(response.headers.get('X-RateLimit-Reset', int(time.time()) + 3600))
            )
        except Exception as e:
            print(f"⚠️  Could not check rate limit: {e}")
            if self.session.headers.get('Authorization'):
                return 5000, 1000, int(time.time()) + 3600
            else:
                return 60, 30, int(time.time()) + 3600
    
    def wait_for_rate_limit(self, reset_time: int):
        """Wait until rate limit resets"""
        current_time = int(time.time())
        wait_time = reset_time - current_time + 5  # Add 5 seconds buffer
        
        if wait_time > 0:
            reset_datetime = datetime.fromtimestamp(reset_time, tz=timezone.utc)
            print(f"\n⏰ Rate limit exceeded. Waiting until {reset_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"   Sleeping for {wait_time} seconds...")
            
            # Show progress while waiting
            for i in range(wait_time, 0, -1):
                print(f"\r   Time remaining: {i} seconds", end='', flush=True)
                time.sleep(1)
            print("\r   Resuming...                    ")
    
    def save_checkpoint(self, checkpoint_file: str, data: Dict):
        """Save checkpoint data"""
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
    
    def load_checkpoint(self, checkpoint_file: str) -> Optional[Dict]:
        """Load checkpoint data if exists"""
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Could not load checkpoint: {e}")
        return None
    
    def get_issues(self, owner: str, repo: str, state: str = 'all', 
                   labels: Optional[List[str]] = None, since: Optional[str] = None,
                   max_issues: Optional[int] = None, resume: bool = True,
                   include_prs: bool = False) -> List[Dict]:
        """
        Fetch issues from a GitHub repository with cursor-based pagination support
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: Issue state ('open', 'closed', 'all')
            labels: List of labels to filter by
            since: Only issues updated after this time (ISO 8601 format)
            max_issues: Maximum number of issues to fetch
            resume: Whether to resume from checkpoint if available
            include_prs: Whether to include pull requests (default: False)
            
        Returns:
            List of issue dictionaries
        """
        # Checkpoint file name
        checkpoint_file = f".checkpoint_{owner}_{repo}_issues_v2.pkl"
        
        # Try to load checkpoint
        issues = []
        next_url = None
        page_count = 0
        
        if resume:
            checkpoint = self.load_checkpoint(checkpoint_file)
            if checkpoint:
                issues = checkpoint.get('issues', [])
                next_url = checkpoint.get('next_url', None)
                page_count = checkpoint.get('page_count', 0)
                print(f"📂 Resuming from checkpoint: {len(issues)} issues already fetched")
                
                # If we already have enough issues, return them
                if max_issues and len(issues) >= max_issues:
                    issues = issues[:max_issues]
                    print(f"✓ Already have {max_issues} issues from checkpoint")
                    return issues
        
        # Build initial URL and parameters
        if not next_url:
            params = {
                'state': state,
                'per_page': 100  # Maximum allowed by GitHub
            }
            
            if labels:
                params['labels'] = ','.join(labels)
            if since:
                params['since'] = since
            
            next_url = f"{self.BASE_URL}/repos/{owner}/{repo}/issues"
            first_request = True
        else:
            params = {}
            first_request = False
        
        print(f"\n📋 Fetching issues from {owner}/{repo}...")
        print(f"   State: {state}")
        if labels:
            print(f"   Labels: {', '.join(labels)}")
        if since:
            print(f"   Since: {since}")
        if include_prs:
            print("   Including pull requests")
        else:
            print("   Excluding pull requests")
        print()
        
        while next_url:
            # Check rate limit before making request
            limit, remaining, reset = self.check_rate_limit()
            
            if remaining <= 1:
                self.wait_for_rate_limit(reset)
            
            # Make request
            page_count += 1
            print(f"📥 Fetching page {page_count}...")
            
            if first_request:
                response = self.session.get(next_url, params=params)
                first_request = False
            else:
                # For subsequent requests, use the full URL from Link header
                response = self.session.get(next_url)
            
            # Handle rate limiting
            if response.status_code == 403 or response.status_code == 429:
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                self.wait_for_rate_limit(reset_time)
                continue
            
            # Handle other errors
            if response.status_code != 200:
                print(f"❌ Error: {response.status_code} - {response.text}")
                break
            
            # Parse issues
            page_issues = response.json()
            
            if not page_issues:
                print("✓ No more issues found")
                break
            
            # Filter out pull requests if requested
            if not include_prs:
                page_issues = [issue for issue in page_issues if 'pull_request' not in issue]
            
            issues.extend(page_issues)
            print(f"✓ Found {len(page_issues)} {'items' if include_prs else 'issues'} (Total: {len(issues)})")
            
            # Parse Link header for next URL
            link_header = response.headers.get('Link', '')
            links = self.parse_link_header(link_header)
            next_url = links.get('next', None)
            
            # Save checkpoint immediately after fetching each page
            checkpoint_data = {
                'issues': issues,
                'next_url': next_url,
                'page_count': page_count,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'state': state,
                'labels': labels,
                'since': since
            }
            self.save_checkpoint(checkpoint_file, checkpoint_data)
            print(f"💾 Checkpoint saved ({len(issues)} items)")
            
            # Check if we've reached the maximum
            if max_issues and len(issues) >= max_issues:
                issues = issues[:max_issues]
                print(f"✓ Reached maximum of {max_issues} items")
                # Clean up checkpoint since we're done
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                    print("🧹 Checkpoint cleaned up")
                break
            
            # Rate limiting delay
            if remaining > 100:
                time.sleep(0.5)  # Fast when we have plenty of requests
            elif remaining > 10:
                time.sleep(1.5)  # Moderate when getting low
            else:
                time.sleep(3)    # Slow when very low
        
        # Clean up checkpoint if we completed successfully
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("🧹 Checkpoint cleaned up (scraping completed)")
        
        return issues
    
    def save_issues(self, issues: List[Dict], output_file: str):
        """Save issues to a JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Count issues vs PRs
        issue_count = sum(1 for item in issues if 'pull_request' not in item)
        pr_count = sum(1 for item in issues if 'pull_request' in item)
        
        # Prepare data with metadata
        data = {
            'metadata': {
                'total_items': len(issues),
                'issues': issue_count,
                'pull_requests': pr_count,
                'scraped_at': datetime.now(timezone.utc).isoformat(),
                'file_version': '2.0'
            },
            'items': issues
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved {len(issues)} items to {output_file}")
        if pr_count > 0:
            print(f"   - Issues: {issue_count}")
            print(f"   - Pull Requests: {pr_count}")
    
    def print_summary(self, issues: List[Dict]):
        """Print a summary of the scraped issues"""
        if not issues:
            print("\n📊 No items found")
            return
        
        # Separate issues and PRs
        real_issues = [item for item in issues if 'pull_request' not in item]
        prs = [item for item in issues if 'pull_request' in item]
        
        # Count by state
        open_issues = sum(1 for issue in real_issues if issue['state'] == 'open')
        closed_issues = sum(1 for issue in real_issues if issue['state'] == 'closed')
        open_prs = sum(1 for pr in prs if pr['state'] == 'open')
        closed_prs = sum(1 for pr in prs if pr['state'] == 'closed')
        
        # Count by labels (issues only)
        label_counts = {}
        for issue in real_issues:
            for label in issue.get('labels', []):
                label_name = label['name']
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
        print("\n📊 Summary:")
        print(f"   Total items: {len(issues)}")
        
        if real_issues:
            print(f"\n   Issues: {len(real_issues)}")
            print(f"   - Open: {open_issues}")
            print(f"   - Closed: {closed_issues}")
        
        if prs:
            print(f"\n   Pull Requests: {len(prs)}")
            print(f"   - Open: {open_prs}")
            print(f"   - Closed: {closed_prs}")
        
        if label_counts:
            print("\n   Top labels (issues only):")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   - {label}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='GitHub Issues Scraper v2 with cursor-based pagination support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape all issues from a repository
  python github_issues_scraper_v2.py facebook/react

  # Include pull requests
  python github_issues_scraper_v2.py facebook/react --include-prs

  # Scrape only closed issues
  python github_issues_scraper_v2.py facebook/react --state closed

  # Resume from checkpoint (automatic)
  python github_issues_scraper_v2.py facebook/react

  # Start fresh, ignore checkpoint
  python github_issues_scraper_v2.py facebook/react --no-resume
        """
    )
    
    parser.add_argument('repository', help='Repository in format owner/repo')
    parser.add_argument('--token', help='GitHub personal access token (for higher rate limits)')
    parser.add_argument('--state', choices=['open', 'closed', 'all'], default='all',
                        help='Issue state to fetch (default: all)')
    parser.add_argument('--labels', help='Comma-separated list of labels to filter by')
    parser.add_argument('--since', help='Only issues updated after this time (ISO 8601 format)')
    parser.add_argument('--max-issues', type=int, help='Maximum number of issues to fetch')
    parser.add_argument('--output', '-o', help='Output JSON file (default: owner_repo_issues.json)')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignore any existing checkpoint')
    parser.add_argument('--include-prs', action='store_true', help='Include pull requests in the results')
    
    args = parser.parse_args()
    
    # Parse repository
    try:
        owner, repo = args.repository.split('/')
    except ValueError:
        print("❌ Error: Repository must be in format owner/repo")
        sys.exit(1)
    
    # Parse labels
    labels = None
    if args.labels:
        labels = [label.strip() for label in args.labels.split(',')]
    
    # Get token from argument or environment variable
    token = args.token or os.environ.get('GITHUB_TOKEN')
    
    # Create scraper
    scraper = GitHubIssuesScraperV2(token)
    
    try:
        # Fetch issues
        issues = scraper.get_issues(
            owner=owner,
            repo=repo,
            state=args.state,
            labels=labels,
            since=args.since,
            max_issues=args.max_issues,
            resume=not args.no_resume,
            include_prs=args.include_prs
        )
        
        # Save to file
        output_file = args.output or f"{owner}_{repo}_issues_v2.json"
        scraper.save_issues(issues, output_file)
        
        # Print summary
        scraper.print_summary(issues)
        
        # Check final rate limit
        limit, remaining, reset = scraper.check_rate_limit()
        reset_time = datetime.fromtimestamp(reset, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"\n📊 Rate limit: {remaining}/{limit} (resets at {reset_time})")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()