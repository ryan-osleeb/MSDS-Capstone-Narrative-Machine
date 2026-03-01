"""
Retail Investor domain — data paths, config key, and scraper.

Data files (CSVs) live in this directory alongside this __init__.py.
"""

from pathlib import Path

DOMAIN_DIR = Path(__file__).parent
DOMAINS_DIR = DOMAIN_DIR.parent
PROJECT_ROOT = DOMAINS_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "retailinvestor"

DOMAIN_MANIFEST = {
    'config_key':       'retail',
    'topic_label':      'retail',
    'nyt_csv':          str(DOMAIN_DIR / 'nyt_retailinvestor.csv'),
    'gdelt_csv':        str(DOMAIN_DIR / 'historical_retailinvestor.csv'),
    'unified_csv':      str(DOMAIN_DIR / 'unified_retailinvestor.csv'),
    'output_prefix':    'retail',
    'cache_prefix':     'retail',
    'sentiment_method': 'textblob',
    'yearly_networks':  [],    # empty = auto-detect from data
}
