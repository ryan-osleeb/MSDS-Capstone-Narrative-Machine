"""
AI/Technology domain — data paths, config key, and scraper.

Data files (CSVs) live in this directory alongside this __init__.py.
"""

from pathlib import Path

DOMAIN_DIR = Path(__file__).parent

DOMAIN_MANIFEST = {
    'config_key':       'aitech',
    'topic_label':      'tech',
    'nyt_csv':          str(DOMAIN_DIR / 'nyt_aitech_articles.csv'),
    'gdelt_csv':        str(DOMAIN_DIR / 'historical_news_tech.csv'),
    'unified_csv':      str(DOMAIN_DIR / 'unified_tech.csv'),
    'output_prefix':    'tech',
    'cache_prefix':     'tech',
    'sentiment_method': 'textblob',
    'yearly_networks':  [],    # empty = auto-detect from data
}