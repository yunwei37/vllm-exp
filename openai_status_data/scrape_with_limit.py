import json
import time
from datetime import datetime
import requests
from urllib.parse import urljoin, urlencode

def fetch_all_incidents_with_pagination(max_pages=50):
    """Fetch all incidents from OpenAI status page API with pagination support"""
    base_url = "https://status.openai.com"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    all_incidents = []
    page = 1
    has_more = True
    empty_page_count = 0
    
    while has_more and page <= max_pages:
        try:
            # Build URL with pagination
            params = {'page': page}
            url = f"{base_url}/api/v2/incidents.json?{urlencode(params)}"
            
            print(f"Fetching page {page}...")
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'incidents' in data and data['incidents']:
                    incidents = data['incidents']
                    print(f"  Found {len(incidents)} incidents on page {page}")
                    all_incidents.extend(incidents)
                    
                    # Reset empty page counter
                    empty_page_count = 0
                    
                    # Check if this looks like the last page
                    if len(incidents) < 20:  # Less than typical page size
                        print(f"  Page {page} has fewer incidents, likely near the end")
                        # Continue for a few more pages to be sure
                    
                    page += 1
                    time.sleep(0.3)  # Be polite between requests
                else:
                    empty_page_count += 1
                    print(f"  No incidents found on page {page}")
                    
                    # Stop if we get 3 empty pages in a row
                    if empty_page_count >= 3:
                        print("  Reached end of data (3 empty pages)")
                        has_more = False
                    else:
                        page += 1
            else:
                print(f"  Failed to fetch page {page}: {response.status_code}")
                has_more = False
                
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            has_more = False
    
    if page > max_pages:
        print(f"\nReached maximum page limit ({max_pages})")
    
    # Also fetch scheduled maintenances
    try:
        maint_url = f"{base_url}/api/v2/scheduled-maintenances.json"
        print(f"\nFetching scheduled maintenances...")
        response = requests.get(maint_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if 'scheduled_maintenances' in data:
                maintenances = data['scheduled_maintenances']
                print(f"  Found {len(maintenances)} scheduled maintenances")
                # Add type field to distinguish from incidents
                for maint in maintenances:
                    maint['type'] = 'scheduled_maintenance'
                all_incidents.extend(maintenances)
    except Exception as e:
        print(f"Error fetching scheduled maintenances: {e}")
    
    return all_incidents

def enrich_incident_data(incidents):
    """Enrich incident data with calculated fields"""
    for incident in incidents:
        # Ensure type field exists
        if 'type' not in incident:
            incident['type'] = 'incident'
        
        # Calculate duration if resolved
        if 'created_at' in incident and 'resolved_at' in incident:
            try:
                created = datetime.fromisoformat(incident['created_at'].replace('Z', '+00:00'))
                resolved = datetime.fromisoformat(incident['resolved_at'].replace('Z', '+00:00'))
                duration = resolved - created
                incident['duration_seconds'] = int(duration.total_seconds())
                incident['duration_human'] = str(duration)
            except:
                pass
        
        # Extract component names
        if 'components' in incident:
            incident['affected_component_names'] = [comp.get('name', '') for comp in incident['components']]
        
        # Count updates
        if 'incident_updates' in incident:
            incident['update_count'] = len(incident['incident_updates'])
            
            # Extract update messages (limit to avoid huge output)
            updates = []
            for update in incident['incident_updates'][:10]:  # Limit to first 10 updates
                if update.get('body'):
                    updates.append({
                        'time': update.get('created_at', ''),
                        'status': update.get('status', ''),
                        'message': update.get('body', '')[:500]  # Limit message length
                    })
            incident['update_messages'] = updates
    
    return incidents

def save_to_json(incidents, filename):
    """Save incidents to JSON file with metadata"""
    output_data = {
        "scrape_date": datetime.now().isoformat(),
        "source": "OpenAI Status Page API",
        "total_incidents": len([i for i in incidents if i.get('type') == 'incident']),
        "total_scheduled_maintenances": len([i for i in incidents if i.get('type') == 'scheduled_maintenance']),
        "total_items": len(incidents),
        "incidents": incidents
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_data

def print_summary(data):
    """Print summary of scraped data"""
    print("\n" + "="*60)
    print("SCRAPING SUMMARY")
    print("="*60)
    print(f"Total items scraped: {data['total_items']}")
    print(f"  - Incidents: {data['total_incidents']}")
    print(f"  - Scheduled maintenances: {data['total_scheduled_maintenances']}")
    
    if data['incidents']:
        # Get date range
        dates = []
        for incident in data['incidents']:
            if 'created_at' in incident:
                dates.append(incident['created_at'])
        
        if dates:
            dates.sort()
            print(f"\nDate range: {dates[0][:10]} to {dates[-1][:10]}")
        
        # Impact summary
        impacts = {}
        for incident in data['incidents']:
            if incident.get('type') == 'incident':
                impact = incident.get('impact', 'unknown')
                impacts[impact] = impacts.get(impact, 0) + 1
        
        if impacts:
            print("\nIncident impact levels:")
            for impact, count in sorted(impacts.items()):
                print(f"  - {impact}: {count}")
        
        # Year summary
        years = {}
        for incident in data['incidents']:
            if 'created_at' in incident:
                year = incident['created_at'][:4]
                years[year] = years.get(year, 0) + 1
        
        if years:
            print("\nIncidents by year:")
            for year, count in sorted(years.items()):
                print(f"  - {year}: {count}")
        
        # Sample incidents
        print("\nMost recent incidents (first 5):")
        recent_incidents = [i for i in data['incidents'] if i.get('type') == 'incident'][:5]
        for i, incident in enumerate(recent_incidents):
            print(f"\n{i+1}. {incident.get('name', 'Unnamed incident')}")
            print(f"   Date: {incident.get('created_at', 'Unknown')[:10]}")
            print(f"   Status: {incident.get('status', 'Unknown')}")
            print(f"   Impact: {incident.get('impact', 'Unknown')}")
            if 'duration_human' in incident:
                print(f"   Duration: {incident['duration_human']}")

if __name__ == "__main__":
    print("OpenAI Status Page Scraper")
    print("=========================\n")
    
    # Fetch all incidents with pagination (limited to prevent timeout)
    print("Fetching incidents from API with pagination...")
    all_incidents = fetch_all_incidents_with_pagination(max_pages=100)
    
    # Enrich the data
    print(f"\nEnriching data for {len(all_incidents)} items...")
    all_incidents = enrich_incident_data(all_incidents)
    
    # Sort by date (newest first)
    all_incidents.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    # Save to JSON
    output_file = "openai_status_data/openai_status_complete.json"
    data = save_to_json(all_incidents, output_file)
    print(f"\nData saved to: {output_file}")
    
    # Print summary
    print_summary(data)