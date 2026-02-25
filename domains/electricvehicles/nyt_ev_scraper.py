"""
NYT Electric Vehicle Targeted Scraper
=====================================
Scrapes NYT articles specifically about electric vehicles.
"""

from nytimes_scraper import NYTScraper
from datetime import datetime, timedelta
import time

# Your API key
API_KEY = "VUzPvbtYRU0NnZ3z7LWaj9VmRVEilT2BJltGRXKHQAoXr4VX"

def scrape_ev_articles():
    scraper = NYTScraper(API_KEY)
    
    # Date range (match your GDELT data: 2015-2024)
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2026, 12, 31)
    
    # Option 1: Broad EV search query
    # The API supports boolean operators
    ev_query = (
        '"electric vehicle" OR "electric vehicles" OR '
        '"electric car" OR "electric cars" OR '
        'EV OR EVs OR '
        'Tesla OR "charging station" OR '
        '"battery electric" OR "zero emission" OR '
        '"plug-in hybrid" OR PHEV'
    )
    
    print("=" * 60)
    print("NYT Electric Vehicle Scraper")
    print("=" * 60)
    print(f"Query: {ev_query[:80]}...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print("=" * 60)
    
    # Run the search
    articles = scraper.scrape_search_date_range(
        query=ev_query,
        start_date=start_date,
        end_date=end_date,
        filter_query=None,  # No section filter - get all sections
        output_file='nyt_ev_articles.csv'
    )
    
    print(f"\n✓ Scraped {len(articles)} EV articles")
    return articles


def scrape_ev_articles_by_section():
    """Alternative: Scrape specific sections for more targeted results"""
    scraper = NYTScraper(API_KEY)
    
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2026, 12, 31)
    
    ev_query = '"electric vehicle" OR Tesla OR EV OR "charging station"'
    
    # Target specific sections
    sections = [
        'Business',
        'Technology', 
        'Climate',
        'Automobiles',
        'Science',
    ]
    
    all_articles = []
    
    for section in sections:
        print(f"\n--- Scraping section: {section} ---")
        filter_query = f'section_name:"{section}"'
        
        try:
            articles = scraper.scrape_search_date_range(
                query=ev_query,
                start_date=start_date,
                end_date=end_date,
                filter_query=filter_query,
                output_file=f'nyt_ev_{section.lower()}.csv'
            )
            all_articles.extend(articles)
            print(f"  Found {len(articles)} articles in {section}")
        except Exception as e:
            print(f"  Error with {section}: {e}")
        
        time.sleep(5)  # Be nice to the API between sections
    
    # Deduplicate by URL
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        if article['web_url'] not in seen_urls:
            seen_urls.add(article['web_url'])
            unique_articles.append(article)
    
    # Save combined
    if unique_articles:
        scraper._save_to_csv(unique_articles, 'nyt_ev_articles_combined.csv')
        print(f"\n✓ Total unique articles: {len(unique_articles)}")
    
    return unique_articles


if __name__ == "__main__":
    # Choose one approach:
    
    # Option 1: Simple broad search (recommended to start)
    articles = scrape_ev_articles()
    
    # Option 2: Section-by-section (more targeted but slower)
    # articles = scrape_ev_articles_by_section()