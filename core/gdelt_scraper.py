"""
core/gdelt_scraper.py — GDELT DocSearch API + article text scraper.

Queries the free GDELT 2.0 DocSearch API for article URLs, then fetches
article text directly from the live web.

Output CSV columns (matches existing historical_news_*.csv schema):
    url, archived, archive_url, archive_timestamp, status,
    original_date, title, text, text_length, success, error, date
"""

import csv
import time
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Max records per GDELT request (hard cap: 250)
MAX_RECORDS = 250
# Slide a weekly window to collect more articles per date range
WINDOW_DAYS = 7
# Delay between article fetch requests (be polite)
FETCH_DELAY = 1.0
# Delay between GDELT API calls
GDELT_DELAY = 2.0
# Max chars to keep from article text
MAX_TEXT_CHARS = 5000
# Minimum content-type check
ARTICLE_CONTENT_TYPES = ("text/html",)

# Known paywall / low-value domains to skip
SKIP_DOMAINS = {
    "wsj.com", "ft.com", "bloomberg.com", "barrons.com",
    "thetimes.co.uk", "economist.com",
}

_JUNK_PATTERNS = re.compile(
    r"(subscribe|sign up|cookie policy|advertisement|"
    r"newsletter|follow us|share this|click here|"
    r"read more|related articles|you might also|"
    r"sponsored|promoted)",
    re.IGNORECASE,
)


class GDELTScraper:
    """
    Fetch recent news articles via GDELT DocSearch, then scrape their text.

    Usage
    -----
    scraper = GDELTScraper()
    scraper.scrape(
        query="electric vehicle OR EV OR Tesla",
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2026, 2, 21),
        output_file="domains/electricvehicles/historical_news_evs.csv",
    )
    """

    def __init__(self, fetch_text: bool = True, fetch_delay: float = FETCH_DELAY):
        self.fetch_text = fetch_text
        self.fetch_delay = fetch_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

    # ── Public interface ───────────────────────────────────────────────────────

    def scrape(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        output_file: str,
        language: str = "English",
    ) -> list[dict]:
        """
        Scrape articles matching *query* between start_date and end_date.

        Slides a weekly window through the date range, deduplicating by URL.
        Appends new rows to output_file if it already exists.
        Returns list of article dicts.
        """
        print(f"\n  Scraping GDELT: {start_date.date()} → {end_date.date()}")
        print(f"  Query: {query[:80]}...")

        existing_urls = self._load_existing_urls(output_file)
        all_articles: list[dict] = []
        seen_urls: set[str] = set(existing_urls)

        window_start = start_date
        while window_start <= end_date:
            window_end = min(window_start + timedelta(days=WINDOW_DAYS - 1), end_date)
            print(f"  Window: {window_start.date()} → {window_end.date()}", flush=True)
            gdelt_articles = self._query_gdelt(query, window_start, window_end, language)

            new_in_window = [
                item for item in gdelt_articles
                if item.get("url") and item["url"] not in seen_urls
                and not self._should_skip(item["url"])
            ]
            print(f"    {len(new_in_window)} new articles to fetch", flush=True)

            for i, item in enumerate(new_in_window, 1):
                url = item["url"]
                seen_urls.add(url)
                row = self._build_row(item)
                if self.fetch_text:
                    print(f"    [{i}/{len(new_in_window)}] {url[:80]}", flush=True)
                    row = self._enrich_with_text(row)
                all_articles.append(row)
                time.sleep(self.fetch_delay)

            window_start = window_end + timedelta(days=1)
            time.sleep(GDELT_DELAY)

        if all_articles:
            self._append_to_csv(all_articles, output_file)
            print(f"  ✓ {len(all_articles)} new articles → {output_file}")
        else:
            print("  ✓ No new articles found")

        return all_articles

    # ── GDELT API ─────────────────────────────────────────────────────────────

    def _query_gdelt(
        self,
        query: str,
        start: datetime,
        end: datetime,
        language: str,
    ) -> list[dict]:
        # GDELT requires OR queries to be wrapped in parentheses
        wrapped = f"({query})" if " OR " in query else query
        params = {
            "query":         f"{wrapped} sourcelang:{language}",
            "mode":          "artlist",
            "maxrecords":    MAX_RECORDS,
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime":   end.strftime("%Y%m%d%H%M%S"),
            "format":        "json",
        }
        for attempt in range(2):
            try:
                resp = self.session.get(GDELT_DOC_URL, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    print(f"  ⚠ GDELT rate limited (429) — waiting {wait}s")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    log.warning("GDELT API returned %s", resp.status_code)
                    return []
                # GDELT returns a 200 with a plain-text error string when the
                # query is too long or too short — detect this before parsing JSON.
                content_type = resp.headers.get("content-type", "")
                if "json" not in content_type:
                    msg = resp.text.strip()[:120]
                    print(f"  ⚠ GDELT non-JSON response (query may be too long): {msg}")
                    log.warning("GDELT plain-text error response: %s", msg)
                    return []
                data = resp.json()
                return data.get("articles", []) or []
            except Exception as exc:
                print(f"  ⚠ GDELT request error: {exc}")
                log.warning("GDELT API error: %s", exc)
                return []
        log.warning("GDELT API still rate-limited after retry — skipping window")
        return []

    # ── Article text fetching ─────────────────────────────────────────────────

    def _enrich_with_text(self, row: dict) -> dict:
        url = row["url"]
        try:
            # (connect_timeout, read_timeout) — prevents hanging on stalled servers
            resp = self.session.get(url, timeout=(5, 15), allow_redirects=True)
            row["status"] = resp.status_code

            if resp.status_code != 200:
                row["success"] = False
                row["error"] = f"HTTP {resp.status_code}"
                return row

            content_type = resp.headers.get("content-type", "")
            if not any(ct in content_type for ct in ARTICLE_CONTENT_TYPES):
                row["success"] = False
                row["error"] = f"Non-HTML content-type: {content_type[:60]}"
                return row

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract title from <title> or <h1> if not already set
            if not row["title"]:
                h1 = soup.find("h1")
                title_tag = soup.find("title")
                row["title"] = (
                    h1.get_text(strip=True) if h1
                    else title_tag.get_text(strip=True) if title_tag
                    else ""
                )

            # Extract body text — prefer <article> tag, fall back to <p> tags
            article_tag = soup.find("article")
            if article_tag:
                paragraphs = article_tag.find_all("p")
            else:
                paragraphs = soup.find_all("p")

            text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
            text = self._clean_text(text)[:MAX_TEXT_CHARS]

            row["text"] = text
            row["text_length"] = len(text)
            row["success"] = len(text) >= 100
            row["error"] = "" if row["success"] else "Too short after cleaning"

        except requests.Timeout:
            row["success"] = False
            row["error"] = "Timeout"
        except Exception as exc:
            row["success"] = False
            row["error"] = str(exc)[:200]

        return row

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_row(item: dict) -> dict:
        url = item.get("url", "")
        seen_date = item.get("seendate", "")  # format: YYYYMMDDTHHMMSSZ
        # Normalise to YYYYMMDD
        date_int = int(seen_date[:8]) if seen_date and seen_date[:8].isdigit() else 0

        return {
            "url":               url,
            "archived":          False,
            "archive_url":       "",
            "archive_timestamp": "",
            "status":            0,
            "original_date":     date_int,
            "title":             item.get("title", ""),
            "text":              "",
            "text_length":       0,
            "success":           False,
            "error":             "",
            "date":              date_int,
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        # Drop lines that are likely navigation / UI chrome
        lines = [
            line for line in text.split(".")
            if len(line.split()) > 6 and not _JUNK_PATTERNS.search(line)
        ]
        return ". ".join(lines).strip()

    @staticmethod
    def _should_skip(url: str) -> bool:
        try:
            # Use netloc directly — the substring check handles both www. and bare domains.
            # lstrip("www.") was a bug: it strips individual chars, so "www.wsj.com"
            # became "sj.com" and wsj.com was never skipped.
            domain = urlparse(url).netloc.lower()
            return any(skip in domain for skip in SKIP_DOMAINS)
        except Exception:
            return False

    @staticmethod
    def _csv_fields() -> list[str]:
        return [
            "url", "archived", "archive_url", "archive_timestamp", "status",
            "original_date", "title", "text", "text_length", "success", "error", "date",
        ]

    def _load_existing_urls(self, output_file: str) -> set[str]:
        path = Path(output_file)
        if not path.exists():
            return set()
        urls: set[str] = set()
        try:
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    url = row.get("url", "")
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
