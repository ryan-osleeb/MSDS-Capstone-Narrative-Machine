"""
core/nytimes_scraper.py — NYT Article Search API wrapper.

Matches the NYTScraper interface expected by the domain scraper scripts
(nyt_ev_scraper.py, nyt_tech_scraper.py).

Output CSV columns (matches existing nyt_*.csv schema):
    date, headline, abstract, snippet, lead_paragraph, web_url,
    section, subsection, byline, document_type, news_desk,
    type_of_material, word_count, source, keywords
"""

import csv
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

NYT_SEARCH_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

# NYT API allows up to 10 pages per date window (100 results).
# For large date ranges we slide a window forward.
MAX_PAGES_PER_WINDOW = 10
RESULTS_PER_PAGE = 10
# Minimum days per sliding window — narrow windows avoid hitting the 10-page cap
WINDOW_DAYS = 30

# Polite delay between requests (NYT allows 5 req/s but 500/day)
REQUEST_DELAY = 0.25


class NYTScraper:
    """Wrapper around the NYT Article Search API v2."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NarrativeMachine/3.0"})

    # ── Public interface ───────────────────────────────────────────────────────

    def scrape_search_date_range(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        filter_query: Optional[str] = None,
        output_file: str = "nyt_articles.csv",
    ) -> list[dict]:
        """
        Scrape all articles matching *query* between start_date and end_date.

        Slides a monthly window to stay within the 10-page per-request cap.
        Deduplicates by URL.  Appends to output_file if it already exists.

        Returns list of article dicts.
        """
        print(f"\n  Scraping NYT: {start_date.date()} → {end_date.date()}")
        print(f"  Query: {query[:80]}...")

        existing_urls = self._load_existing_urls(output_file)
        all_articles: list[dict] = []
        seen_urls: set[str] = set(existing_urls)

        window_start = start_date
        while window_start <= end_date:
            window_end = min(window_start + timedelta(days=WINDOW_DAYS - 1), end_date)
            window_articles = self._scrape_window(
                query, window_start, window_end, filter_query, seen_urls
            )
            all_articles.extend(window_articles)
            window_start = window_end + timedelta(days=1)

        if all_articles:
            self._append_to_csv(all_articles, output_file)
            print(f"  ✓ {len(all_articles)} new articles → {output_file}")
        else:
            print(f"  ✓ No new articles found")

        return all_articles

    def _save_to_csv(self, articles: list[dict], output_file: str) -> None:
        """Save articles to CSV (overwrites). Used by domain scrapers directly."""
        if not articles:
            return
        path = Path(output_file)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields())
            writer.writeheader()
            writer.writerows(articles)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _scrape_window(
        self,
        query: str,
        start: datetime,
        end: datetime,
        filter_query: Optional[str],
        seen_urls: set[str],
    ) -> list[dict]:
        articles: list[dict] = []
        begin_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        for page in range(MAX_PAGES_PER_WINDOW):
            params = {
                "q": query,
                "begin_date": begin_str,
                "end_date": end_str,
                "page": page,
                "api-key": self.api_key,
                "sort": "oldest",
            }
            if filter_query:
                params["fq"] = filter_query

            try:
                resp = self.session.get(NYT_SEARCH_URL, params=params, timeout=30)
                time.sleep(REQUEST_DELAY)

                if resp.status_code == 429:
                    print("    ⚠ Rate limited — waiting 60 s")
                    time.sleep(60)
                    resp = self.session.get(NYT_SEARCH_URL, params=params, timeout=30)

                if resp.status_code != 200:
                    log.warning("HTTP %s for window %s page %s", resp.status_code, begin_str, page)
                    break

                data = resp.json()
                docs = data.get("response", {}).get("docs", [])
                if not docs:
                    break

                for doc in docs:
                    url = doc.get("web_url", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    articles.append(self._parse_doc(doc))

                total_hits = data.get("response", {}).get("meta", {}).get("hits", 0)
                fetched_so_far = (page + 1) * RESULTS_PER_PAGE
                if fetched_so_far >= total_hits:
                    break

            except requests.RequestException as exc:
                log.warning("Request error: %s", exc)
                time.sleep(5)
                break

        return articles

    def _parse_doc(self, doc: dict) -> dict:
        keywords = ", ".join(
            kw.get("value", "") for kw in doc.get("keywords", []) if kw.get("value")
        )
        byline_raw = doc.get("byline") or {}
        byline = byline_raw.get("original", "") if isinstance(byline_raw, dict) else ""

        return {
            "date":             doc.get("pub_date", ""),
            "headline":         (doc.get("headline") or {}).get("main", ""),
            "abstract":         doc.get("abstract", ""),
            "snippet":          doc.get("snippet", ""),
            "lead_paragraph":   doc.get("lead_paragraph", ""),
            "web_url":          doc.get("web_url", ""),
            "section":          doc.get("section_name", ""),
            "subsection":       doc.get("subsection_name", ""),
            "byline":           byline,
            "document_type":    doc.get("document_type", ""),
            "news_desk":        doc.get("news_desk", ""),
            "type_of_material": doc.get("type_of_material", ""),
            "word_count":       doc.get("word_count", 0),
            "source":           doc.get("source", ""),
            "keywords":         keywords,
        }

    @staticmethod
    def _csv_fields() -> list[str]:
        return [
            "date", "headline", "abstract", "snippet", "lead_paragraph",
            "web_url", "section", "subsection", "byline", "document_type",
            "news_desk", "type_of_material", "word_count", "source", "keywords",
        ]

    def _load_existing_urls(self, output_file: str) -> set[str]:
        path = Path(output_file)
        if not path.exists():
            return set()
        urls: set[str] = set()
        try:
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    url = row.get("web_url", "")
                    if url:
                        urls.add(url)
        except Exception:
            pass
        return urls

    def _append_to_csv(self, articles: list[dict], output_file: str) -> None:
        path = Path(output_file)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields())
            if write_header:
                writer.writeheader()
            writer.writerows(articles)
