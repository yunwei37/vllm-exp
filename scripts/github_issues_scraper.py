#!/usr/bin/env python3
"""
GitHub Issues Scraper with Rate Limiting

This script fetches all issues from a given GitHub repository with proper rate limiting
and error handling. It supports both authenticated and unauthenticated requests.
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

# Load environment variables from .env file
load_dotenv()


class GitHubIssuesScraper:
    """Scraper for GitHub issues with rate limiting support"""
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the scraper
        
        Args:
            token: GitHub personal access token (optional)
        """
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'GitHub-Issues-Scraper/1.0'
        })
        
        if token:
            self.session.headers.update({'Authorization': f'token {token}'})
            print("✓ Using authenticated requests (5,000 requests/hour)")
        else:
            print("⚠ Using unauthenticated requests (60 requests/hour)")
    
    def check_rate_limit(self) -> Tuple[int, int, int]:
        """
        Check current rate limit status
        
        Returns:
            Tuple of (limit, remaining, reset_timestamp)
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/rate_limit")
            
            if response.status_code == 200:
                data = response.json()
                # Handle different response structures
                if 'rate' in data and 'core' in data['rate']:
                    core = data['rate']['core']
                    return core['limit'], core['remaining'], core['reset']
                elif 'core' in data:
                    core = data['core']
                    return core['limit'], core['remaining'], core['reset']
                else:
                    # Fallback to headers
                    return (
                        int(response.headers.get('X-RateLimit-Limit', 5000 if self.session.headers.get('Authorization') else 60)),
                        int(response.headers.get('X-RateLimit-Remaining', 50)),
                        int(response.headers.get('X-RateLimit-Reset', int(time.time()) + 3600))
                    )
            else:
                # If we can't check rate limit, return from headers
                return (
                    int(response.headers.get('X-RateLimit-Limit', 5000 if self.session.headers.get('Authorization') else 60)),
                    int(response.headers.get('X-RateLimit-Remaining', 50)),
                    int(response.headers.get('X-RateLimit-Reset', int(time.time()) + 3600))
                )
        except Exception as e:
            # Default values based on authentication status
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
                   max_issues: Optional[int] = None, resume: bool = True) -> List[Dict]:
        """
        Fetch issues from a GitHub repository
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: Issue state ('open', 'closed', 'all')
            labels: List of labels to filter by
            since: Only issues updated after this time (ISO 8601 format)
            max_issues: Maximum number of issues to fetch
            resume: Whether to resume from checkpoint if available
            
        Returns:
            List of issue dictionaries
        """
        # Checkpoint file name
        checkpoint_file = f".checkpoint_{owner}_{repo}_issues.pkl"
        
        # Try to load checkpoint
        issues = []
        page = 1
        per_page = 100  # Maximum allowed by GitHub
        
        if resume:
            checkpoint = self.load_checkpoint(checkpoint_file)
            if checkpoint:
                issues = checkpoint.get('issues', [])
                page = checkpoint.get('page', 1)
                print(f"📂 Resuming from checkpoint: {len(issues)} issues already fetched, starting from page {page}")
                
                # If we already have enough issues, return them
                if max_issues and len(issues) >= max_issues:
                    issues = issues[:max_issues]
                    print(f"✓ Already have {max_issues} issues from checkpoint")
                    return issues
        
        # Build query parameters
        params = {
            'state': state,
            'per_page': per_page,
            'page': page
        }
        
        if labels:
            params['labels'] = ','.join(labels)
        if since:
            params['since'] = since
        
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/issues"
        
        print(f"\n📋 Fetching issues from {owner}/{repo}...")
        print(f"   State: {state}")
        if labels:
            print(f"   Labels: {', '.join(labels)}")
        if since:
            print(f"   Since: {since}")
        print()
        
        while True:
            # Check rate limit before making request
            limit, remaining, reset = self.check_rate_limit()
            
            if remaining <= 1:
                self.wait_for_rate_limit(reset)
            
            # Make request
            print(f"📥 Fetching page {page} (up to {per_page} issues)...")
            response = self.session.get(url, params=params)
            
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
            
            # Filter out pull requests (they're included in issues endpoint)
            page_issues = [issue for issue in page_issues if 'pull_request' not in issue]
            
            issues.extend(page_issues)
            print(f"✓ Found {len(page_issues)} issues (Total: {len(issues)})")
            
            # Save checkpoint immediately after fetching each page
            checkpoint_data = {
                'issues': issues,
                'page': page + 1,  # Next page to fetch
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'state': state,
                'labels': labels,
                'since': since
            }
            self.save_checkpoint(checkpoint_file, checkpoint_data)
            print(f"💾 Checkpoint saved (page {page}, {len(issues)} issues)")
            
            # Check if we've reached the maximum
            if max_issues and len(issues) >= max_issues:
                issues = issues[:max_issues]
                print(f"✓ Reached maximum of {max_issues} issues")
                # Clean up checkpoint since we're done
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                    print("🧹 Checkpoint cleaned up")
                break
            
            # Check if there are more pages
            if 'Link' in response.headers:
                if 'rel="next"' not in response.headers['Link']:
                    print("✓ Reached last page")
                    break
            else:
                # No Link header means this is the last page
                break
            
            # Update page number
            page += 1
            params['page'] = page
            
            # Rate limiting delay
            # For authenticated requests: ~1.4 seconds between requests
            # For unauthenticated requests: ~60 seconds between requests
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
        
        # Prepare data with metadata
        data = {
            'metadata': {
                'total_issues': len(issues),
                'scraped_at': datetime.now(timezone.utc).isoformat(),
                'file_version': '1.0'
            },
            'issues': issues
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved {len(issues)} issues to {output_file}")
    
    def print_summary(self, issues: List[Dict]):
        """Print a summary of the scraped issues"""
        if not issues:
            print("\n📊 No issues found")
            return
        
        # Count by state
        open_count = sum(1 for issue in issues if issue['state'] == 'open')
        closed_count = sum(1 for issue in issues if issue['state'] == 'closed')
        
        # Count by labels
        label_counts = {}
        for issue in issues:
            for label in issue.get('labels', []):
                label_name = label['name']
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
        print("\n📊 Summary:")
        print(f"   Total issues: {len(issues)}")
        print(f"   Open: {open_count}")
        print(f"   Closed: {closed_count}")
        
        if label_counts:
            print("\n   Top labels:")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   - {label}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Scrape GitHub issues with rate limiting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape all issues from a repository
  python github_issues_scraper.py facebook/react

  # Scrape only open issues
  python github_issues_scraper.py facebook/react --state open

  # Scrape with authentication (higher rate limit)
  python github_issues_scraper.py facebook/react --token YOUR_GITHUB_TOKEN

  # Scrape issues with specific labels
  python github_issues_scraper.py facebook/react --labels "bug,help wanted"

  # Scrape recent issues (last 30 days)
  python github_issues_scraper.py facebook/react --since 2024-01-01T00:00:00Z

  # Limit number of issues
  python github_issues_scraper.py facebook/react --max-issues 100

  # Resume from checkpoint (automatic)
  python github_issues_scraper.py facebook/react

  # Start fresh, ignore checkpoint
  python github_issues_scraper.py facebook/react --no-resume
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
    scraper = GitHubIssuesScraper(token)
    
    try:
        # Fetch issues
        issues = scraper.get_issues(
            owner=owner,
            repo=repo,
            state=args.state,
            labels=labels,
            since=args.since,
            max_issues=args.max_issues,
            resume=not args.no_resume
        )
        
        # Save to file
        output_file = args.output or f"{owner}_{repo}_issues.json"
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
        sys.exit(1)


if __name__ == "__main__":
    main()