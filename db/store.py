"""
db/store.py — SQLite persistence layer for Narrative Machine.

Tables:
    articles      — canonical unified article records (one row per article)
    gdelt_events  — enriched GDELT events with narrative matching scores
"""

import sqlite3
from pathlib import Path

import pandas as pd

# Default DB location: project root
DB_PATH = Path(__file__).parent.parent / "narrative_machine.db"

# ─── Schema ───────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS articles (
    doc_id              TEXT,
    domain_folder       TEXT    NOT NULL,
    source_type         TEXT,
    domain              TEXT,
    outlet              TEXT,
    published_at        TEXT,
    url                 TEXT,
    title               TEXT,
    section             TEXT,
    full_text           TEXT,
    snippet             TEXT,
    topic_label         TEXT,
    language            TEXT,
    text_len            INTEGER,
    extraction_success  INTEGER,
    duplicate_group_id  TEXT,
    gdelt_metadata      TEXT,
    PRIMARY KEY (doc_id, domain_folder)
);

CREATE TABLE IF NOT EXISTS gdelt_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    date                INTEGER,
    url                 TEXT,
    matched_narratives  TEXT,
    event_code          INTEGER,
    quad_class          INTEGER,
    sentiment           REAL,
    impact              REAL,
    num_mentions        INTEGER,
    actor1_name         TEXT,
    actor2_name         TEXT
);

CREATE INDEX IF NOT EXISTS idx_articles_domain_folder ON articles (domain_folder);
CREATE INDEX IF NOT EXISTS idx_articles_published_at  ON articles (published_at);
CREATE INDEX IF NOT EXISTS idx_articles_topic_label   ON articles (topic_label);
CREATE INDEX IF NOT EXISTS idx_articles_source_type   ON articles (source_type);
CREATE INDEX IF NOT EXISTS idx_gdelt_events_date      ON gdelt_events (date);
"""

# Columns that exist in both the unified CSV and the articles table
_ARTICLE_COLS = [
    "doc_id", "source_type", "domain", "outlet", "published_at", "url",
    "title", "section", "full_text", "snippet", "topic_label", "language",
    "text_len", "extraction_success", "duplicate_group_id", "gdelt_metadata",
]

# Mapping from gdelt_events.csv column names → DB column names
_GDELT_COL_MAP = {
    "date":               "date",
    "url":                "url",
    "matched_narratives": "matched_narratives",
    "EventCode":          "event_code",
    "QuadClass":          "quad_class",
    "sentiment":          "sentiment",
    "impact":             "impact",
    "NumMentions":        "num_mentions",
    "Actor1Name":         "actor1_name",
    "Actor2Name":         "actor2_name",
}


# ─── Initialisation ───────────────────────────────────────────────────────────

def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables and indexes if they don't already exist."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_DDL)


# ─── Write ────────────────────────────────────────────────────────────────────

def upsert_articles(df: pd.DataFrame, domain_folder: str,
                    db_path: Path = DB_PATH) -> int:
    """
    Insert or replace articles from a unified DataFrame.

    Uses INSERT OR REPLACE so re-running ingest is idempotent (doc_id is the
    primary key).  Returns the number of rows written.
    """
    init_db(db_path)

    rows = []
    for _, row in df.iterrows():
        record = {}
        for col in _ARTICLE_COLS:
            val = row.get(col)
            if col == "extraction_success":
                val = None if pd.isna(val) else int(bool(val))
            elif pd.isna(val):
                val = None
            record[col] = val
        record["domain_folder"] = domain_folder
        rows.append(list(record.values()))

    all_cols = _ARTICLE_COLS + ["domain_folder"]
    placeholders = ", ".join("?" * len(all_cols))
    sql = f"INSERT OR REPLACE INTO articles ({', '.join(all_cols)}) VALUES ({placeholders})"

    with sqlite3.connect(db_path) as conn:
        conn.executemany(sql, rows)

    return len(rows)


def upsert_gdelt_events(df: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    """
    Replace all rows in gdelt_events with the supplied DataFrame.

    The table is cleared first because gdelt_events.csv is a global file
    (not domain-specific) and always represents the full current dataset.
    Returns the number of rows written.
    """
    init_db(db_path)

    db_cols = list(_GDELT_COL_MAP.values())
    rows = []
    for _, row in df.iterrows():
        record = []
        for src_col in _GDELT_COL_MAP:
            val = row.get(src_col)
            record.append(None if pd.isna(val) else val)
        rows.append(record)

    placeholders = ", ".join("?" * len(db_cols))
    sql = f"INSERT INTO gdelt_events ({', '.join(db_cols)}) VALUES ({placeholders})"

    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM gdelt_events")
        conn.executemany(sql, rows)

    return len(rows)


# ─── Read ─────────────────────────────────────────────────────────────────────

def get_articles_df(domain_folder: str, db_path: Path = DB_PATH) -> pd.DataFrame | None:
    """
    Return all articles for a domain as a DataFrame, sorted by published_at.
    Returns None if the database does not exist or the domain has no rows.
    """
    if not db_path.exists():
        return None

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT * FROM articles WHERE domain_folder = ? ORDER BY published_at",
            conn,
            params=(domain_folder,),
        )

    if df.empty:
        return None

    df["extraction_success"] = df["extraction_success"].astype(bool)
    return df


def get_db_stats(db_path: Path = DB_PATH) -> dict:
    """
    Return a dict keyed by domain_folder with per-domain statistics:
        total, extracted, earliest, latest, sources {source_type: count}
    Returns an empty dict if the database does not exist.
    """
    if not db_path.exists():
        return {}

    with sqlite3.connect(db_path) as conn:
        domain_rows = conn.execute("""
            SELECT
                domain_folder,
                COUNT(*)               AS total,
                SUM(extraction_success) AS extracted,
                MIN(published_at)      AS earliest,
                MAX(published_at)      AS latest
            FROM articles
            GROUP BY domain_folder
            ORDER BY domain_folder
        """).fetchall()

        source_rows = conn.execute("""
            SELECT domain_folder, source_type, COUNT(*) AS n
            FROM articles
            GROUP BY domain_folder, source_type
        """).fetchall()

    stats: dict = {}
    for domain_folder, total, extracted, earliest, latest in domain_rows:
        stats[domain_folder] = {
            "total":     total,
            "extracted": int(extracted or 0),
            "earliest":  earliest,
            "latest":    latest,
            "sources":   {},
        }

    for domain_folder, source_type, n in source_rows:
        if domain_folder in stats:
            stats[domain_folder]["sources"][source_type] = n

    return stats
