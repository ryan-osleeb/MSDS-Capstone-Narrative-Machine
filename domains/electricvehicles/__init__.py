"""
Electric Vehicles domain — data paths, config key, and scraper.

Data files (CSVs) live in this directory alongside this __init__.py.
"""

from pathlib import Path

DOMAIN_DIR = Path(__file__).parent

DOMAIN_MANIFEST = {
    'config_key':       'ev',
    'topic_label':      'EV',
    'nyt_csv':          str(DOMAIN_DIR / 'nyt_ev_articles.csv'),
    'gdelt_csv':        str(DOMAIN_DIR / 'historical_news_evs.csv'),
    'unified_csv':      str(DOMAIN_DIR / 'unified_ev.csv'),
    'output_prefix':    'ev',
    'cache_prefix':     'ev',
    'sentiment_method': 'textblob',
    'yearly_networks':  [],    # empty = auto-detect from data
}