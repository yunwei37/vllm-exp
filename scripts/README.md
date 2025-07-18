# GitHub Issues Scraper

A Python script to scrape GitHub issues from any repository with proper rate limiting, authentication support, and comprehensive filtering options.

## Features

- 🚦 **Smart Rate Limiting**: Automatically handles GitHub API rate limits with intelligent delays
- 🔐 **Authentication Support**: Works with or without GitHub personal access tokens
- 🎯 **Flexible Filtering**: Filter by state, labels, and date
- 📊 **Progress Tracking**: Real-time progress updates and summaries
- 💾 **JSON Export**: Saves issues in structured JSON format
- ⏸️ **Graceful Interruption**: Can be safely interrupted and respects rate limits
- 🔄 **Resume Support**: Automatically saves checkpoints and resumes from where it left off

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Scrape all issues from a repository:

```bash
python github_issues_scraper.py owner/repository
```

Example:
```bash
python github_issues_scraper.py facebook/react
```

### With Authentication (Recommended)

For higher rate limits (5,000 vs 60 requests/hour), use a GitHub personal access token:

```bash
# Option 1: Pass token as argument
python github_issues_scraper.py owner/repo --token YOUR_GITHUB_TOKEN

# Option 2: Set environment variable
export GITHUB_TOKEN=your_github_personal_access_token
python github_issues_scraper.py owner/repo
```

### Filtering Options

#### By State
```bash
# Only open issues
python github_issues_scraper.py owner/repo --state open

# Only closed issues
python github_issues_scraper.py owner/repo --state closed

# All issues (default)
python github_issues_scraper.py owner/repo --state all
```

#### By Labels
```bash
# Single label
python github_issues_scraper.py owner/repo --labels "bug"

# Multiple labels (comma-separated)
python github_issues_scraper.py owner/repo --labels "bug,help wanted,good first issue"
```

#### By Date
```bash
# Issues updated after a specific date (ISO 8601 format)
python github_issues_scraper.py owner/repo --since 2024-01-01T00:00:00Z
```

#### Limit Number of Issues
```bash
# Fetch only the first 100 issues
python github_issues_scraper.py owner/repo --max-issues 100
```

### Output Options

By default, issues are saved to `owner_repo_issues.json`. You can specify a custom output file:

```bash
python github_issues_scraper.py owner/repo --output my_issues.json
```

### Resume/Checkpoint Support

The scraper automatically saves progress after fetching each page of issues. If interrupted (e.g., rate limit, network error, Ctrl+C), it can resume from where it left off:

```bash
# Start scraping (automatically creates checkpoints)
python github_issues_scraper.py large-org/huge-repo

# If interrupted, simply run the same command to resume
python github_issues_scraper.py large-org/huge-repo
# Output: 📂 Resuming from checkpoint: 500 issues already fetched, starting from page 6

# To start fresh and ignore existing checkpoint
python github_issues_scraper.py large-org/huge-repo --no-resume
```

**How it works:**
- Checkpoints are saved as `.checkpoint_owner_repo_issues.pkl` files
- Each checkpoint contains all fetched issues and the next page to fetch
- Checkpoints are automatically cleaned up when scraping completes successfully
- Checkpoints include metadata like timestamp, filters used, etc.

## Examples

### Example 1: Scrape Recent Bugs
```bash
python github_issues_scraper.py microsoft/vscode \
    --state open \
    --labels "bug,verified" \
    --since 2024-01-01T00:00:00Z \
    --output vscode_recent_bugs.json
```

### Example 2: Quick Sample
```bash
# Get 50 most recent issues for testing
python github_issues_scraper.py pytorch/pytorch \
    --max-issues 50 \
    --output pytorch_sample.json
```

### Example 3: Large Repository with Resume
```bash
# Scraping a large repository (may take time and hit rate limits)
python github_issues_scraper.py kubernetes/kubernetes --max-issues 1000

# If interrupted, resume from checkpoint
python github_issues_scraper.py kubernetes/kubernetes --max-issues 1000
# The scraper will automatically continue from where it stopped
```

### Example 4: Using the Example Script
```bash
# Run all example scenarios
python example_usage.py
```

## Output Format

The script saves issues in the following JSON structure:

```json
{
  "metadata": {
    "total_issues": 150,
    "scraped_at": "2024-01-15T10:30:00Z",
    "file_version": "1.0"
  },
  "issues": [
    {
      "number": 123,
      "title": "Issue title",
      "state": "open",
      "created_at": "2024-01-10T08:00:00Z",
      "updated_at": "2024-01-14T15:30:00Z",
      "labels": [...],
      "user": {...},
      "comments": 5,
      ...
    }
  ]
}
```

## Rate Limiting

### Limits
- **Unauthenticated**: 60 requests per hour
- **Authenticated**: 5,000 requests per hour
- **GitHub Enterprise Cloud**: 15,000 requests per hour

### Automatic Handling
The script automatically:
- Monitors rate limit status via API headers
- Implements progressive delays based on remaining requests
- Waits when rate limit is exceeded with countdown timer
- Shows current rate limit status in real-time

### Rate Limit Strategy
- **> 100 requests remaining**: 0.5 second delay
- **10-100 requests remaining**: 1.5 second delay  
- **< 10 requests remaining**: 3 second delay
- **0 requests remaining**: Waits until reset time

## Getting a GitHub Token

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Click "Generate new token (classic)"
3. Give it a descriptive name
4. Select scopes: `repo` (for private repos) or `public_repo` (for public repos only)
5. Generate and copy the token

## Troubleshooting

### Rate Limit Exceeded
- Use authentication for higher limits
- The script will automatically wait and resume
- Check your current rate limit status:
  ```bash
  curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/rate_limit
  ```

### Connection Errors
- Check your internet connection
- Verify the repository exists and is accessible
- For private repositories, ensure your token has appropriate permissions

### Large Repositories
- Use `--max-issues` to limit the number of issues
- Filter by state or labels to reduce the dataset
- Consider using `--since` to get only recent issues

## Advanced Usage

### Processing Scraped Data

```python
import json

# Load scraped issues
with open('repo_issues.json', 'r') as f:
    data = json.load(f)
    issues = data['issues']

# Example: Find issues by keyword
keyword_issues = [
    issue for issue in issues 
    if 'bug' in issue['title'].lower()
]

# Example: Group by milestone
by_milestone = {}
for issue in issues:
    milestone = issue.get('milestone', {}).get('title', 'No milestone')
    by_milestone.setdefault(milestone, []).append(issue)
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This script is provided as-is for educational and research purposes. Please respect GitHub's Terms of Service and API usage guidelines.