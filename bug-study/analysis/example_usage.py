#!/usr/bin/env python3
"""
Example usage of the GitHub Issues Scraper
"""

from github_issues_scraper import GitHubIssuesScraper
import json
from datetime import datetime, timedelta, timezone

def example_basic_usage():
    """Basic example: Scrape all issues from a small repository"""
    print("=" * 60)
    print("Example 1: Basic usage - Scraping a small repository")
    print("=" * 60)
    
    # Create scraper without authentication (60 requests/hour limit)
    scraper = GitHubIssuesScraper()
    
    # Fetch issues from a small repository
    issues = scraper.get_issues(
        owner="octocat",
        repo="Hello-World",
        state="all"
    )
    
    # Save to file
    scraper.save_issues(issues, "output/hello_world_issues.json")
    scraper.print_summary(issues)
    
    print("\n✓ Basic example completed!")


def example_authenticated_usage():
    """Example with authentication for higher rate limits"""
    print("\n" + "=" * 60)
    print("Example 2: Authenticated usage")
    print("=" * 60)
    
    # You can set the GITHUB_TOKEN environment variable or pass it directly
    # export GITHUB_TOKEN=your_token_here
    import os
    token = os.environ.get('GITHUB_TOKEN')
    
    if not token:
        print("⚠️  No GITHUB_TOKEN found. Set it as environment variable for this example.")
        print("   export GITHUB_TOKEN=your_github_personal_access_token")
        return
    
    # Create authenticated scraper (5,000 requests/hour limit)
    scraper = GitHubIssuesScraper(token=token)
    
    # Fetch recent issues with specific labels
    issues = scraper.get_issues(
        owner="microsoft",
        repo="vscode",
        state="open",
        labels=["bug", "verified"],
        max_issues=50  # Limit to 50 issues for this example
    )
    
    scraper.save_issues(issues, "output/vscode_bugs.json")
    scraper.print_summary(issues)
    
    print("\n✓ Authenticated example completed!")


def example_filtered_scraping():
    """Example with various filters"""
    print("\n" + "=" * 60)
    print("Example 3: Filtered scraping - Recent issues only")
    print("=" * 60)
    
    scraper = GitHubIssuesScraper()
    
    # Calculate date 7 days ago
    since_date = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    
    # Fetch only recent issues
    issues = scraper.get_issues(
        owner="python",
        repo="cpython",
        state="open",
        since=since_date,
        max_issues=20
    )
    
    scraper.save_issues(issues, "output/cpython_recent_issues.json")
    scraper.print_summary(issues)
    
    print("\n✓ Filtered example completed!")


def example_process_issues():
    """Example of processing scraped issues"""
    print("\n" + "=" * 60)
    print("Example 4: Processing scraped issues")
    print("=" * 60)
    
    # Load previously scraped issues
    try:
        with open("output/hello_world_issues.json", 'r') as f:
            data = json.load(f)
            issues = data['issues']
    except FileNotFoundError:
        print("⚠️  No scraped data found. Run example_basic_usage() first.")
        return
    
    print(f"\n📋 Processing {len(issues)} issues:")
    
    # Example: Find issues by a specific user
    user_issues = [
        issue for issue in issues 
        if issue['user']['login'] == 'octocat'
    ]
    print(f"\n   Issues by octocat: {len(user_issues)}")
    
    # Example: Find issues with most comments
    sorted_by_comments = sorted(issues, key=lambda x: x['comments'], reverse=True)
    print("\n   Top 5 issues by comment count:")
    for issue in sorted_by_comments[:5]:
        print(f"   - #{issue['number']}: {issue['title']} ({issue['comments']} comments)")
    
    # Example: Group by labels
    label_groups = {}
    for issue in issues:
        for label in issue['labels']:
            label_name = label['name']
            if label_name not in label_groups:
                label_groups[label_name] = []
            label_groups[label_name].append(issue['number'])
    
    print("\n   Issues grouped by label:")
    for label, issue_numbers in sorted(label_groups.items()):
        print(f"   - {label}: {len(issue_numbers)} issues")
    
    print("\n✓ Processing example completed!")


def main():
    """Run all examples"""
    print("GitHub Issues Scraper - Examples")
    print("================================\n")
    
    # Create output directory
    import os
    os.makedirs("output", exist_ok=True)
    
    # Run examples
    try:
        example_basic_usage()
        example_authenticated_usage()
        example_filtered_scraping()
        example_process_issues()
        
        print("\n\n✅ All examples completed successfully!")
        print("\nCheck the 'output' directory for generated JSON files.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")


if __name__ == "__main__":
    main()