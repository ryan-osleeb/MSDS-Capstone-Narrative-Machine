"""
Template domain — copy this folder to create a new analysis domain.

Steps to add a new domain:
  1. Copy this folder:          cp -r domains/template domains/my_domain
  2. Edit __init__.py:          update DOMAIN_MANIFEST with your paths/keys
  3. Add a config:              add MY_CONFIG to core/narrative_config.py
                                and register it in CONFIG_REGISTRY
  4. Add your data:             place CSVs in domains/my_domain/
  5. (Optional) Add a scraper:  create domains/my_domain/scraper.py
  6. Run:                       python run_domain.py --domain my_domain
"""

from pathlib import Path

DOMAIN_DIR = Path(__file__).parent

DOMAIN_MANIFEST = {
    # ── REQUIRED: update all of these ──────────────────────────────────
    'config_key':       'CHANGE_ME',           # key in CONFIG_REGISTRY (narrative_config.py)
    'topic_label':      'CHANGE_ME',           # label for CanonicalDatasetBuilder
    'nyt_csv':          str(DOMAIN_DIR / 'nyt_CHANGE_ME.csv'),
    'gdelt_csv':        str(DOMAIN_DIR / 'historical_news_CHANGE_ME.csv'),
    'unified_csv':      str(DOMAIN_DIR / 'unified_CHANGE_ME.csv'),
    'output_prefix':    'CHANGE_ME',           # prefix for output filenames
    'cache_prefix':     'CHANGE_ME',           # prefix for embedding caches

    # ── OPTIONAL ──────────────────────────────────────────────────────
    'sentiment_method': 'textblob',            # 'textblob' or 'transformer'
    'yearly_networks':  [],                    # specific years or [] for auto
}