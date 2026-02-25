"""
NYT AI/Technology Targeted Scraper
==================================
Scrapes NYT articles specifically about artificial intelligence and technology.
"""

from nytimes_scraper import NYTScraper
from datetime import datetime
import time

# Your API key
API_KEY = "VUzPvbtYRU0NnZ3z7LWaj9VmRVEilT2BJltGRXKHQAoXr4VX"


def scrape_aitech_articles():
    """Broad search for AI/Tech articles covering all narrative themes."""
    scraper = NYTScraper(API_KEY)
    
    # Date range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2026, 12, 31)
    
    # Comprehensive AI/Tech query covering key narrative themes
    # Progress & Innovation, Labor, Social Harm, Governance, Infrastructure, Finance, Risk, Geopolitics, Inequality
    aitech_query = (
        # Core AI terms
        '"artificial intelligence" OR "machine learning" OR '
        '"deep learning" OR "neural network" OR '
        'ChatGPT OR GPT-4 OR "large language model" OR '
        '"generative AI" OR OpenAI OR Anthropic OR '
        
        # Major players
        'Google AI OR "Google DeepMind" OR Microsoft AI OR '
        'Meta AI OR "AI startup" OR '
        
        # Labor displacement
        '"AI automation" OR "job automation" OR "robots replacing" OR '
        
        # Safety and governance
        '"AI safety" OR "AI regulation" OR "AI ethics" OR '
        '"AI alignment" OR "existential risk" OR '
        
        # Geopolitics
        '"AI chips" OR "semiconductor" OR NVIDIA OR '
        '"AI arms race" OR "tech cold war"'
    )
    
    print("=" * 60)
    print("NYT AI/Technology Scraper")
    print("=" * 60)
    print(f"Query: {aitech_query[:100]}...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print("=" * 60)
    
    # Run the search
    articles = scraper.scrape_search_date_range(
        query=aitech_query,
        start_date=start_date,
        end_date=end_date,
        filter_query=None,
        output_file='nyt_aitech_articles.csv'
    )
    
    print(f"\n✓ Scraped {len(articles)} AI/Tech articles")
    return articles


def scrape_aitech_articles_by_section():
    """Section-by-section scraping for more targeted results."""
    scraper = NYTScraper(API_KEY)
    
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Focused query for section-based scraping
    aitech_query = (
        '"artificial intelligence" OR "machine learning" OR '
        'ChatGPT OR OpenAI OR "generative AI" OR '
        '"AI regulation" OR "AI safety" OR NVIDIA'
    )
    
    # Target sections most relevant to AI/Tech narratives
    sections = [
        'Technology',
        'Business',
        'Science',
        'Opinion',
        'U.S.',
        'World',
        'Magazine',
    ]
    
    all_articles = []
    
    for section in sections:
        print(f"\n--- Scraping section: {section} ---")
        filter_query = f'section_name:"{section}"'
        
        try:
            articles = scraper.scrape_search_date_range(
                query=aitech_query,
                start_date=start_date,
                end_date=end_date,
                filter_query=filter_query,
                output_file=f'nyt_aitech_{section.lower().replace(".", "")}.csv'
            )
            all_articles.extend(articles)
            print(f"  Found {len(articles)} articles in {section}")
        except Exception as e:
            print(f"  Error with {section}: {e}")
        
        time.sleep(5)  # Rate limiting between sections
    
    # Deduplicate by URL
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        if article['web_url'] not in seen_urls:
            seen_urls.add(article['web_url'])
            unique_articles.append(article)
    
    # Save combined
    if unique_articles:
        scraper._save_to_csv(unique_articles, 'nyt_aitech_articles_combined.csv')
        print(f"\n✓ Total unique articles: {len(unique_articles)}")
    
    return unique_articles


def scrape_aitech_by_narrative_theme():
    """
    Scrape articles by narrative theme for more granular analysis.
    Useful if you want separate datasets per narrative.
    """
    scraper = NYTScraper(API_KEY)
    
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Queries tailored to each narrative theme
    narrative_queries = {
        'progress_innovation': (
            '"AI breakthrough" OR "AI innovation" OR "machine learning advances" OR '
            '"AI capabilities" OR "AI assistant" OR "AI productivity" OR '
            '"deep learning research" OR "AI transforms"'
        ),
        'labor_displacement': (
            '"AI jobs" OR "automation jobs" OR "AI replacing workers" OR '
            '"robots replacing" OR "AI unemployment" OR "AI workforce" OR '
            '"automation threat" OR "AI labor"'
        ),
        'social_cultural_harm': (
            '"AI misinformation" OR "deepfake" OR "AI bias" OR '
            '"algorithmic bias" OR "AI discrimination" OR "AI privacy" OR '
            '"AI surveillance" OR "social media algorithm"'
        ),
        'governance_regulation': (
            '"AI regulation" OR "AI legislation" OR "AI policy" OR '
            '"tech regulation" OR "AI governance" OR "AI ethics" OR '
            '"AI antitrust" OR "AI compliance"'
        ),
        'infrastructure_limits': (
            '"AI energy" OR "data center" OR "AI computing" OR '
            '"GPU shortage" OR "AI infrastructure" OR "chip shortage" OR '
            '"AI sustainability" OR "AI power consumption"'
        ),
        'finance_speculation': (
            '"AI stocks" OR "AI valuation" OR "AI investment" OR '
            '"AI bubble" OR "AI IPO" OR "AI venture capital" OR '
            'NVIDIA stock OR "AI market cap"'
        ),
        'existential_risk': (
            '"AI existential risk" OR "superintelligent AI" OR "AI safety" OR '
            '"AI alignment" OR "AI catastrophe" OR "AGI risk" OR '
            '"pause AI" OR "AI doom"'
        ),
        'geopolitics_security': (
            '"AI China" OR "AI arms race" OR "AI military" OR '
            '"AI export controls" OR "AI national security" OR '
            '"AI cyber" OR "AI warfare" OR "tech cold war"'
        ),
        'inequality_concentration': (
            '"AI inequality" OR "AI digital divide" OR "big tech AI" OR '
            '"AI monopoly" OR "AI concentration" OR "AI winners losers" OR '
            '"tech giants AI"'
        ),
    }
    
    all_articles = []
    
    for narrative, query in narrative_queries.items():
        print(f"\n--- Scraping narrative: {narrative} ---")
        
        try:
            articles = scraper.scrape_search_date_range(
                query=query,
                start_date=start_date,
                end_date=end_date,
                filter_query=None,
                output_file=f'nyt_aitech_{narrative}.csv'
            )
            
            # Tag articles with narrative source
            for article in articles:
                article['search_narrative'] = narrative
            
            all_articles.extend(articles)
            print(f"  Found {len(articles)} articles for {narrative}")
        except Exception as e:
            print(f"  Error with {narrative}: {e}")
        
        time.sleep(5)  # Rate limiting
    
    # Deduplicate by URL (keeping first occurrence)
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        if article['web_url'] not in seen_urls:
            seen_urls.add(article['web_url'])
            unique_articles.append(article)
    
    if unique_articles:
        scraper._save_to_csv(unique_articles, 'nyt_aitech_by_narrative.csv')
        print(f"\n✓ Total unique articles: {len(unique_articles)}")
    
    return unique_articles


if __name__ == "__main__":
    # Choose one approach:
    
    # Option 1: Broad comprehensive search (recommended to start)
    articles = scrape_aitech_articles()
    
    # Option 2: Section-by-section (more targeted)
    # articles = scrape_aitech_articles_by_section()
    
    # Option 3: By narrative theme (for pre-labeled data)
    # articles = scrape_aitech_by_narrative_theme()