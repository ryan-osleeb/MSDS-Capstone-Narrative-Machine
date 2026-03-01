"""
domains/retailinvestor/wayback_retailinvestor_scraper.py

Two-step pipeline to build historical_retailinvestor.csv:

  Step 1 — migrate_legacy_data()
    Converts legacy/historical_news_retail.csv (32 MB, ~235 k rows,
    schema has error_x/error_y) to the current 12-column schema and writes
    domains/retailinvestor/historical_retailinvestor.csv.  No HTTP calls.

  Step 2 — scrape_remaining(limit=None)
    Reads legacy/gdelt_events_retail.csv for the full URL+date list, skips
    URLs already in the output file, and fetches missing articles via the
    Wayback Machine (archive.org).  Appends one row per URL immediately
    (crash-safe / resumable).

Output schema (matches GDELTScraper._csv_fields() exactly):
    url, archived, archive_url, archive_timestamp, status, original_date,
    title, text, text_length, success, error, date
"""

import csv
import re
import time
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────

_HERE           = Path(__file__).parent
_PROJECT_ROOT   = _HERE.parent.parent
_LEGACY_DIR     = _PROJECT_ROOT / "legacy"

LEGACY_GDELT_CSV   = _LEGACY_DIR / "gdelt_events_retail.csv"
LEGACY_HISTORY_CSV = _LEGACY_DIR / "historical_news_retail.csv"
OUTPUT_FILE        = _HERE / "historical_retailinvestor.csv"

# ── Timing ─────────────────────────────────────────────────────────────────

WAYBACK_API_DELAY = 1.5   # seconds between Wayback availability checks
FETCH_DELAY       = 2.0   # seconds between article fetches
MAX_TEXT_CHARS    = 5000
MIN_TEXT_LEN      = 200   # success threshold

WAYBACK_API = "https://archive.org/wayback/available"

# ── Junk-line filter (mirrors GDELTScraper) ────────────────────────────────

_JUNK_PATTERNS = re.compile(
    r"(subscribe|sign up|cookie policy|advertisement|"
    r"newsletter|follow us|share this|click here|"
    r"read more|related articles|you might also|"
    r"sponsored|promoted)",
    re.IGNORECASE,
)


# ── Schema helpers ─────────────────────────────────────────────────────────

def _csv_fields() -> list[str]:
    """12-column schema matching GDELTScraper._csv_fields()."""
    return [
        "url", "archived", "archive_url", "archive_timestamp", "status",
        "original_date", "title", "text", "text_length", "success", "error", "date",
    ]


def _load_existing_urls(output_file: Path) -> set[str]:
    """Return set of URLs already in *output_file* (idempotency guard)."""
    if not output_file.exists():
        return set()
    urls: set[str] = set()
    try:
        with output_file.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                url = row.get("url", "").strip()
                if url:
                    urls.add(url)
    except Exception as exc:
        log.warning("Could not read existing URLs from %s: %s", output_file, exc)
    return urls


def _append_to_csv(rows: list[dict], output_file: Path) -> None:
    """Append *rows* to *output_file*, writing header only when file is new or empty."""
    write_header = not output_file.exists() or output_file.stat().st_size == 0
    with output_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fields())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ── Text-cleaning helper (mirrors GDELTScraper._clean_text) ───────────────

def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    lines = [
        line for line in text.split(".")
        if len(line.split()) > 6 and not _JUNK_PATTERNS.search(line)
    ]
    return ". ".join(lines).strip()


# ── Step 1: migrate legacy data ────────────────────────────────────────────

def migrate_legacy_data(
    legacy_csv: Path = LEGACY_HISTORY_CSV,
    output_file: Path = OUTPUT_FILE,
) -> int:
    """
    Convert legacy/historical_news_retail.csv → current 12-column schema.

    The legacy file has error_x / error_y instead of a single error column.
    This function merges them (prefers whichever is non-empty) and skips URLs
    already present in *output_file*.

    Returns the number of rows written.
    """
    if not legacy_csv.exists():
        print(f"  ⚠ Legacy CSV not found: {legacy_csv}")
        return 0

    existing_urls = _load_existing_urls(output_file)
    print(f"  migrate_legacy_data: {len(existing_urls)} URLs already in output")

    fields = _csv_fields()
    batch: list[dict] = []
    written = 0
    skipped = 0
    BATCH_SIZE = 5000

    with legacy_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            url = row.get("url", "").strip()
            if not url or url in existing_urls:
                skipped += 1
                continue

            existing_urls.add(url)  # prevent duplicates within this run

            # Merge error_x / error_y → single error field
            error_x = (row.get("error_x") or "").strip()
            error_y = (row.get("error_y") or "").strip()
            error = error_x if error_x else error_y

            out_row = {
                "url":               url,
                "archived":          row.get("archived", ""),
                "archive_url":       row.get("archive_url", ""),
                "archive_timestamp": row.get("archive_timestamp", ""),
                "status":            row.get("status", ""),
                "original_date":     row.get("original_date", ""),
                "title":             row.get("title", ""),
                "text":              row.get("text", ""),
                "text_length":       row.get("text_length", ""),
                "success":           row.get("success", ""),
                "error":             error,
                "date":              row.get("date", ""),
            }

            batch.append(out_row)
            if len(batch) >= BATCH_SIZE:
                _append_to_csv(batch, output_file)
                written += len(batch)
                print(f"    ... {written:,} rows written so far", flush=True)
                batch = []

    if batch:
        _append_to_csv(batch, output_file)
        written += len(batch)

    print(f"  ✓ migrate_legacy_data: {written:,} rows written, {skipped:,} skipped")
    return written


# ── Step 2: scrape remaining URLs via Wayback Machine ─────────────────────

def _wayback_check(session: requests.Session, url: str, date_str: str) -> dict:
    """
    Query the Wayback availability API for *url* near *date_str* (YYYYMMDD).

    Returns a dict with keys: archived, archive_url, archive_timestamp, status.
    """
    result = {
        "archived":          False,
        "archive_url":       "",
        "archive_timestamp": "",
        "status":            0,
    }
    try:
        resp = session.get(
            WAYBACK_API,
            params={"url": url, "timestamp": date_str},
            timeout=15,
        )
        result["status"] = resp.status_code
        if resp.status_code != 200:
            return result
        data = resp.json()
        snapshot = (data.get("archived_snapshots") or {}).get("closest") or {}
        if snapshot.get("available"):
            result["archived"]          = True
            result["archive_url"]       = snapshot.get("url", "")
            result["archive_timestamp"] = snapshot.get("timestamp", "")
            result["status"]            = int(snapshot.get("status", 0) or 0)
    except Exception as exc:
        log.debug("Wayback API error for %s: %s", url, exc)
    return result


def _fetch_article_text(session: requests.Session, archive_url: str) -> dict:
    """
    Fetch and parse article text from an archived Wayback URL.

    Returns a dict with keys: title, text, text_length, success, error.
    """
    result = {"title": "", "text": "", "text_length": 0, "success": False, "error": ""}
    try:
        resp = session.get(archive_url, timeout=(8, 20), allow_redirects=True)
        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}"
            return result

        soup = BeautifulSoup(resp.text, "html.parser")

        # Strip Wayback Machine toolbar elements
        for toolbar_id in ("wm-ipp-base", "donato"):
            tag = soup.find(id=toolbar_id)
            if tag:
                tag.decompose()

        # Strip noise tags
        for tag_name in ("script", "style", "nav", "header", "footer", "aside"):
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Title
        h1 = soup.find("h1")
        title_tag = soup.find("title")
        title = (
            h1.get_text(strip=True) if h1
            else title_tag.get_text(strip=True) if title_tag
            else ""
        )

        # Body text — prefer <article>, fall back to all <p>
        article_tag = soup.find("article")
        paragraphs = article_tag.find_all("p") if article_tag else soup.find_all("p")
        raw_text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        text = _clean_text(raw_text)[:MAX_TEXT_CHARS]

        result["title"]       = title
        result["text"]        = text
        result["text_length"] = len(text)
        result["success"]     = len(text) >= MIN_TEXT_LEN
        result["error"]       = "" if result["success"] else "Too short after cleaning"

    except requests.Timeout:
        result["error"] = "Timeout"
    except Exception as exc:
        result["error"] = str(exc)[:200]

    return result


def scrape_remaining(
    gdelt_csv: Path = LEGACY_GDELT_CSV,
    output_file: Path = OUTPUT_FILE,
    limit: int | None = None,
) -> int:
    """
    Scrape URLs from *gdelt_csv* that are not yet in *output_file*.

    For each URL:
      1. Check the Wayback Machine availability API.
      2. If archived, fetch article text from the archived snapshot.
      3. Append one row to *output_file* immediately (crash-safe).

    Pass *limit* to process only the first N unscraped URLs (useful for testing).
    Returns the number of rows written.
    """
    if not gdelt_csv.exists():
        print(f"  ⚠ GDELT events CSV not found: {gdelt_csv}")
        return 0

    existing_urls = _load_existing_urls(output_file)
    print(f"  scrape_remaining: {len(existing_urls)} URLs already done")

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })

    written = 0
    skipped = 0

    with gdelt_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, gdelt_row in enumerate(reader):
            if limit is not None and written >= limit:
                break

            url      = gdelt_row.get("url", "").strip()
            date_str = str(gdelt_row.get("date", "")).strip()

            if not url:
                continue
            if url in existing_urls:
                skipped += 1
                continue

            existing_urls.add(url)  # prevent re-processing within this run

            print(f"  [{written + 1}] {url[:90]}", flush=True)

            # Build base row
            out_row: dict = {
                "url":               url,
                "archived":          False,
                "archive_url":       "",
                "archive_timestamp": "",
                "status":            0,
                "original_date":     date_str,
                "title":             "",
                "text":              "",
                "text_length":       0,
                "success":           False,
                "error":             "",
                "date":              date_str,
            }

            # Step 1: Wayback availability check
            wb = _wayback_check(session, url, date_str)
            time.sleep(WAYBACK_API_DELAY)

            out_row["archived"]          = wb["archived"]
            out_row["archive_url"]       = wb["archive_url"]
            out_row["archive_timestamp"] = wb["archive_timestamp"]
            out_row["status"]            = wb["status"]

            if not wb["archived"]:
                out_row["error"] = "Not archived"
                _append_to_csv([out_row], output_file)
                written += 1
                continue

            # Step 2: Fetch article text
            article = _fetch_article_text(session, wb["archive_url"])
            time.sleep(FETCH_DELAY)

            out_row["title"]       = article["title"]
            out_row["text"]        = article["text"]
            out_row["text_length"] = article["text_length"]
            out_row["success"]     = article["success"]
            out_row["error"]       = article["error"]

            _append_to_csv([out_row], output_file)
            written += 1

    print(f"  ✓ scrape_remaining: {written:,} rows written, {skipped:,} already done")
    return written


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    print("=== Step 1: Migrate legacy Wayback data ===")
    migrate_legacy_data()

    print("\n=== Step 2: Scrape remaining URLs via Wayback Machine ===")
    scrape_remaining()
