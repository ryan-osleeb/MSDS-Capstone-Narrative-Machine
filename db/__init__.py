"""db — SQLite persistence layer for Narrative Machine."""

from .store import (
    DB_PATH,
    init_db,
    upsert_articles,
    upsert_gdelt_events,
    get_articles_df,
    get_db_stats,
)

__all__ = [
    "DB_PATH",
    "init_db",
    "upsert_articles",
    "upsert_gdelt_events",
    "get_articles_df",
    "get_db_stats",
]
