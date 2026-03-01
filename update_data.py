#!/usr/bin/env python3
"""
update_data.py — Fetch new articles and update the database.

Pulls only articles published AFTER the latest date already stored for each
domain, so it's safe to re-run at any time without creating duplicates.

Usage:
    # Update all domains
    ./narrative_machine/bin/python update_data.py

    # Update a specific domain
    ./narrative_machine/bin/python update_data.py --domain aitech

    # Fetch from NYT only (skip GDELT)
    ./narrative_machine/bin/python update_data.py --sources nyt

    # Fetch from GDELT only
    ./narrative_machine/bin/python update_data.py --sources gdelt

    # After fetching, also re-run the full pipeline
    ./narrative_machine/bin/python update_data.py --run-pipeline
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

# ── Project path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core"))

load_dotenv(PROJECT_ROOT / ".env")

from core.nytimes_scraper import NYTScraper
from core.gdelt_scraper import GDELTScraper
from db import get_db_stats, upsert_articles
from domains import list_available_domains, get_domain_manifest

# ── Domain scraper configurations ────────────────────────────────────────────

NYT_API_KEY = os.environ.get("NYT_API_KEY")
if not NYT_API_KEY:
    raise EnvironmentError("NYT_API_KEY not set. Add it to your .env file or environment.")

# Each entry: domain_folder → {nyt: {query, output_file}, gdelt: {query, output_file}}
DOMAIN_SCRAPER_CONFIG = {
    "electricvehicles": {
        "nyt": {
            "query": (
                '"electric vehicle" OR "electric vehicles" OR '
                '"electric car" OR "electric cars" OR '
                'EV OR EVs OR Tesla OR "charging station" OR '
                '"battery electric" OR "zero emission" OR '
                '"plug-in hybrid" OR PHEV'
            ),
            "output_file": str(PROJECT_ROOT / "domains/electricvehicles/nyt_ev_articles.csv"),
        },
        "gdelt": {
            "query": (
                '"electric vehicle" OR "electric car" OR '
                'Tesla OR EV OR "charging station" OR '
                '"battery electric" OR "zero emission"'
            ),
            "output_file": str(PROJECT_ROOT / "domains/electricvehicles/historical_news_evs.csv"),
        },
    },
    "aitech": {
        "nyt": {
            "query": (
                '"artificial intelligence" OR "machine learning" OR '
                '"deep learning" OR "neural network" OR '
                'ChatGPT OR GPT-4 OR "large language model" OR '
                '"generative AI" OR OpenAI OR Anthropic OR '
                '"AI automation" OR "job automation" OR '
                '"AI safety" OR "AI regulation" OR "AI ethics" OR '
                '"AI chips" OR semiconductor OR NVIDIA OR '
                '"AI arms race" OR "tech cold war"'
            ),
            "output_file": str(PROJECT_ROOT / "domains/aitech/nyt_aitech_articles.csv"),
        },
        "gdelt": {
            "query": (
                '"artificial intelligence" OR "machine learning" OR '
                'ChatGPT OR OpenAI OR "generative AI" OR '
                '"AI regulation" OR "AI safety" OR NVIDIA OR '
                '"large language model" OR "AI chip"'
            ),
            "output_file": str(PROJECT_ROOT / "domains/aitech/historical_news_tech.csv"),
        },
    },
    "retailinvestor": {
        "nyt": {
            # Short query — the NYT API silently returns docs=null when a query
            # chains many quoted phrases with OR.  Unquoted words and individual
            # quoted phrases work reliably.  The bulk scraper
            # (domains/retailinvestor/nyt_retailinvestor_scraper.py) runs multiple
            # sequential sub-queries to populate the initial dataset; this entry
            # handles incremental updates going forward.
            "query": (
                'GameStop OR Robinhood OR WallStreetBets OR '
                '"retail investor" OR "short squeeze" OR "meme stock" OR '
                '"day trading" OR "index fund"'
            ),
            "output_file": str(PROJECT_ROOT / "domains/retailinvestor/nyt_retailinvestor.csv"),
        },
        "gdelt": {
            # Kept short — GDELT returns a plain-text error (not JSON) when the
            # query exceeds ~250 chars, which the scraper previously mistook for
            # a request exception and silently returned 0 articles.
            "query": (
                'GameStop OR Robinhood OR WallStreetBets OR '
                '"retail investor" OR "short squeeze" OR "meme stock" OR '
                '"day trading" OR "index fund"'
            ),
            "output_file": str(PROJECT_ROOT / "domains/retailinvestor/historical_retailinvestor.csv"),
        },
    },
}

# Earliest start date to use when a domain has NO data in the DB yet
DOMAIN_EARLIEST_START = {
    "electricvehicles": datetime(2015, 1, 1),
    "aitech":           datetime(2020, 1, 1),
    "retailinvestor":   datetime(2019, 1, 1),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_fetch_start(domain_folder: str, stats: dict) -> datetime:
    """
    Return the date to start fetching from.
    If the domain already has data, start from the day after the latest article.
    Otherwise use the domain's configured earliest start date.
    """
    domain_stats = stats.get(domain_folder)
    if domain_stats and domain_stats.get("latest"):
        latest_str = domain_stats["latest"][:10]  # "YYYY-MM-DD"
        try:
            latest_dt = datetime.strptime(latest_str, "%Y-%m-%d")
            return latest_dt + timedelta(days=1)
        except ValueError:
            pass
    return DOMAIN_EARLIEST_START.get(domain_folder, datetime(2020, 1, 1))


def run_ingest(domain_folder: str) -> None:
    """Re-run the ingest step to rebuild the unified CSV and update the DB."""
    import run_domain
    from core import get_config

    print(f"\n  Re-running ingest for {domain_folder}...")
    manifest = get_domain_manifest(domain_folder)
    output_dir = PROJECT_ROOT / "output" / domain_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    run_domain.step_ingest(manifest, output_dir, domain_folder)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch new articles and update the Narrative Machine database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--domain", "-d",
        default=None,
        help="Domain to update (default: all available domains)",
    )
    parser.add_argument(
        "--sources", "-s",
        default="nyt,gdelt",
        help="Comma-separated sources to fetch: nyt, gdelt (default: nyt,gdelt)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date for fetching, YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="After fetching, re-run the full analysis pipeline for updated domains",
    )
    args = parser.parse_args()

    sources = [s.strip().lower() for s in args.sources.split(",")]
    end_date = (
        datetime.strptime(args.end_date, "%Y-%m-%d")
        if args.end_date
        else datetime.today().replace(hour=23, minute=59, second=59)
    )

    domains = [args.domain] if args.domain else list(DOMAIN_SCRAPER_CONFIG.keys())
    invalid = [d for d in domains if d not in DOMAIN_SCRAPER_CONFIG]
    if invalid:
        print(f"Unknown domains: {invalid}. Available: {list(DOMAIN_SCRAPER_CONFIG.keys())}")
        sys.exit(1)

    print("=" * 60)
    print("Narrative Machine — Data Update")
    print("=" * 60)
    print(f"Domains:  {domains}")
    print(f"Sources:  {sources}")
    print(f"End date: {end_date.date()}")

    db_stats = get_db_stats()
    updated_domains = []

    for domain_folder in domains:
        cfg = DOMAIN_SCRAPER_CONFIG[domain_folder]
        start_date = get_fetch_start(domain_folder, db_stats)

        if start_date >= end_date:
            print(f"\n── {domain_folder} ──")
            print(f"  Already up to date (latest: {db_stats.get(domain_folder, {}).get('latest', 'none')})")
            continue

        print(f"\n{'='*60}")
        print(f"Domain: {domain_folder}")
        print(f"Fetch window: {start_date.date()} → {end_date.date()}")
        print("=" * 60)

        fetched_any = False

        if "nyt" in sources:
            nyt_cfg = cfg["nyt"]
            scraper = NYTScraper(NYT_API_KEY)
            articles = scraper.scrape_search_date_range(
                query=nyt_cfg["query"],
                start_date=start_date,
                end_date=end_date,
                output_file=nyt_cfg["output_file"],
            )
            if articles:
                fetched_any = True

        if "gdelt" in sources:
            gdelt_cfg = cfg["gdelt"]
            scraper = GDELTScraper()
            articles = scraper.scrape(
                query=gdelt_cfg["query"],
                start_date=start_date,
                end_date=end_date,
                output_file=gdelt_cfg["output_file"],
            )
            if articles:
                fetched_any = True

        if fetched_any:
            run_ingest(domain_folder)
            updated_domains.append(domain_folder)
        else:
            print(f"  No new articles fetched for {domain_folder}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Update complete")
    print("=" * 60)
    new_stats = get_db_stats()
    for d in domains:
        old = db_stats.get(d, {}).get("total", 0)
        new = new_stats.get(d, {}).get("total", 0)
        latest = new_stats.get(d, {}).get("latest", "?")[:10]
        print(f"  {d}: {old} → {new} articles  (latest: {latest})")

    if args.run_pipeline and updated_domains:
        print(f"\nRunning pipeline for: {updated_domains}")
        import subprocess, os
        python = str(PROJECT_ROOT / "narrative_machine" / "bin" / "python")
        env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
        for d in updated_domains:
            subprocess.run(
                [python, "run_domain.py", "--domain", d,
                 "--steps", "analyze,viz,network,extensions,ext_viz"],
                cwd=str(PROJECT_ROOT),
                env=env,
            )


if __name__ == "__main__":
    main()
