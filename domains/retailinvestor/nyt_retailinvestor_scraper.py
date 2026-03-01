"""
NYT Retail Investor Bulk Scraper
=================================
Scrapes NYT articles covering retail investor narratives.

Root cause of the original 0-article problem:
    The NYT Article Search API silently returns docs=null when a query
    combines multiple quoted phrases with OR (e.g. "retail investor" OR
    "meme stock" OR ...).  Simple, short queries (single quoted phrase
    or a few unquoted words) work reliably.

Fix: run multiple sequential single-term queries against the same
output file.  NYTScraper._load_existing_urls() deduplicates
automatically on each call, so re-running is safe.

Checkpoint: completed sub-queries are recorded in nyt_scraper_checkpoint.json
so re-runs skip already-finished work and don't waste the 500 req/day limit.
Use --reset to clear the checkpoint and start fresh.

Usage (from project root):
    ./narrative_machine/bin/python domains/retailinvestor/nyt_retailinvestor_scraper.py
    ./narrative_machine/bin/python domains/retailinvestor/nyt_retailinvestor_scraper.py --reset
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from dotenv import load_dotenv
import os

load_dotenv(PROJECT_ROOT / ".env")

from core.nytimes_scraper import NYTScraper

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FILE    = str(PROJECT_ROOT / "domains" / "retailinvestor" / "nyt_retailinvestor.csv")
CHECKPOINT_FILE = PROJECT_ROOT / "domains" / "retailinvestor" / "nyt_scraper_checkpoint.json"

START_DATE = datetime(2019, 1, 1)
END_DATE   = datetime.today()

# Sub-queries: each is intentionally short so the NYT API returns real results.
# Quoted phrases work fine individually; OR-chaining quoted phrases causes
# the API to silently return docs=null.
SUB_QUERIES = [
    # Meme stocks / platforms
    ("meme_platforms",   "GameStop OR Robinhood OR WallStreetBets"),
    # Short selling / volatility
    ("short_squeeze",    '"short squeeze"'),
    # Retail investor identity
    ("retail_investor",  '"retail investor"'),
    # Passive investing
    ("index_fund",       '"index fund"'),
    # Day / active trading
    ("day_trading",      '"day trading"'),
    # Meme stock culture
    ("meme_stock",       '"meme stock"'),
    # Personal finance angle
    ("passive_investing", '"passive investing"'),
    # Inflation / crypto retail angle
    ("inflation_hedge",  '"inflation hedge"'),
]

# Section-based filter queries (used by scrape_by_section)
TARGET_SECTIONS = ["Business", "Opinion", "Technology", "Your Money"]


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint() -> set[str]:
    """Return the set of sub-query labels that have already been completed."""
    if not CHECKPOINT_FILE.exists():
        return set()
    try:
        data = json.loads(CHECKPOINT_FILE.read_text())
        return set(data.get("completed", []))
    except Exception:
        return set()


def mark_complete(label: str) -> None:
    """Append label to the checkpoint file."""
    completed = load_checkpoint()
    completed.add(label)
    CHECKPOINT_FILE.write_text(json.dumps({"completed": sorted(completed)}, indent=2))


def reset_checkpoint() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print(f"Checkpoint cleared: {CHECKPOINT_FILE}")
    else:
        print("No checkpoint file found — nothing to reset.")


# ── Scrapers ──────────────────────────────────────────────────────────────────

def scrape_retailinvestor_articles() -> list[dict]:
    """
    Broad bulk scrape: runs each sub-query sequentially against the same
    output file.  Skips sub-queries already recorded in the checkpoint.
    Deduplication is handled by NYTScraper automatically.
    """
    api_key = os.environ.get("NYT_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "NYT_API_KEY not set. Add it to your .env file or environment."
        )

    scraper = NYTScraper(api_key)
    completed = load_checkpoint()

    pending = [(label, q) for label, q in SUB_QUERIES if label not in completed]
    skipped = [label for label, _ in SUB_QUERIES if label in completed]

    print("=" * 60)
    print("NYT Retail Investor Scraper — Broad Search")
    print("=" * 60)
    print(f"Date range:  {START_DATE.date()} → {END_DATE.date()}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Sub-queries: {len(pending)} to run, {len(skipped)} already done")
    if skipped:
        print(f"Skipping:    {', '.join(skipped)}")
    print("=" * 60)

    all_new: list[dict] = []

    for label, query in pending:
        print(f"\n── Sub-query: {label} ──")
        articles = scraper.scrape_search_date_range(
            query=query,
            start_date=START_DATE,
            end_date=END_DATE,
            filter_query=None,
            output_file=OUTPUT_FILE,
        )
        all_new.extend(articles)
        mark_complete(label)
        # Brief pause between query batches (NYT daily limit is 500 req/day)
        time.sleep(2)

    print(f"\n{'='*60}")
    print(f"Broad scrape complete — {len(all_new)} total new articles written")
    print("=" * 60)
    return all_new


def scrape_by_section() -> list[dict]:
    """
    Optional: same queries filtered to high-signal NYT sections.
    Useful if the broad search returns too much noise.
    """
    api_key = os.environ.get("NYT_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "NYT_API_KEY not set. Add it to your .env file or environment."
        )

    scraper = NYTScraper(api_key)

    print("=" * 60)
    print("NYT Retail Investor Scraper — Section-Filtered Search")
    print("=" * 60)

    all_new: list[dict] = []

    for section in TARGET_SECTIONS:
        fq = f'section_name:"{section}"'
        for label, query in SUB_QUERIES:
            print(f"\n── {section} / {label} ──")
            articles = scraper.scrape_search_date_range(
                query=query,
                start_date=START_DATE,
                end_date=END_DATE,
                filter_query=fq,
                output_file=OUTPUT_FILE,
            )
            all_new.extend(articles)
            time.sleep(2)

    print(f"\n{'='*60}")
    print(f"Section scrape complete — {len(all_new)} total new articles written")
    print("=" * 60)
    return all_new


if __name__ == "__main__":
    if "--reset" in sys.argv:
        reset_checkpoint()
        sys.exit(0)

    # Default: broad search across all sections
    scrape_retailinvestor_articles()

    # Optional: section-filtered (uncomment to run after broad search)
    # scrape_by_section()
