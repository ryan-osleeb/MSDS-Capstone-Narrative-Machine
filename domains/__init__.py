"""
domains — Domain-specific configurations, scrapers, and data paths.

Each sub-package must expose a DOMAIN_MANIFEST dict with:
    config_key      : str   — key for narrative_config.get_config()
    topic_label     : str   — label passed to CanonicalDatasetBuilder
    nyt_csv         : str   — filename of NYT scraped articles (inside domain dir)
    gdelt_csv       : str   — filename of GDELT data (inside domain dir)
    unified_csv     : str   — filename for the unified output
    output_prefix   : str   — prefix for visualization filenames
    cache_prefix    : str   — prefix for embedding/model caches
    sentiment_method: str   — 'textblob' or 'transformer'
    yearly_networks : list  — specific years to render, or empty for auto
"""

import importlib
from pathlib import Path

# Auto-discover domain packages (skip __pycache__, template, and files)
_DOMAINS_DIR = Path(__file__).parent
AVAILABLE_DOMAINS = {}

for child in sorted(_DOMAINS_DIR.iterdir()):
    if child.is_dir() and child.name not in ('__pycache__', 'template', '_template'):
        try:
            mod = importlib.import_module(f'domains.{child.name}')
            if hasattr(mod, 'DOMAIN_MANIFEST'):
                AVAILABLE_DOMAINS[child.name] = mod.DOMAIN_MANIFEST
        except Exception:
            pass  # skip domains that fail to import


def get_domain_manifest(name: str) -> dict:
    """Look up a domain manifest by folder name."""
    if name not in AVAILABLE_DOMAINS:
        raise ValueError(
            f"Unknown domain '{name}'. "
            f"Available: {list(AVAILABLE_DOMAINS.keys())}"
        )
    return AVAILABLE_DOMAINS[name]


def list_available_domains() -> list:
    """Return list of discovered domain names."""
    return list(AVAILABLE_DOMAINS.keys())