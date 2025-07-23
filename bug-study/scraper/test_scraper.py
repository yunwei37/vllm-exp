#!/usr/bin/env python3
"""
Quick test of the GitHub Issues Scraper
"""

from github_issues_scraper import GitHubIssuesScraper
import os

def test_basic_scraping():
    """Test basic scraping functionality"""
    print("Testing GitHub Issues Scraper")
    print("=" * 40)
    
    # Create scraper
    token = os.environ.get('GITHUB_TOKEN')
    scraper = GitHubIssuesScraper(token)
    
    # Test with a small repository
    print("\nTesting with octocat/Hello-World (small repo)...")
    issues = scraper.get_issues(
        owner="octocat",
        repo="Hello-World",
        max_issues=5  # Just get 5 issues for testing
    )
    
    if issues:
        print(f"\n✅ Success! Found {len(issues)} issues")
        print("\nFirst issue:")
        print(f"  #{issues[0]['number']}: {issues[0]['title']}")
        print(f"  State: {issues[0]['state']}")
        print(f"  Created: {issues[0]['created_at']}")
    else:
        print("\n⚠️  No issues found (this repo might not have issues)")
    
    # Check rate limit
    limit, remaining, reset = scraper.check_rate_limit()
    print(f"\n📊 Rate limit status: {remaining}/{limit}")

if __name__ == "__main__":
    test_basic_scraping()