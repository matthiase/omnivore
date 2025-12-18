# ============================================================================
# FILE: src/omnivore/db.py
# ============================================================================
import psycopg
from contextlib import contextmanager
from omnivore.config import config


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = psycopg.connect(config.database_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_cursor():
    """Context manager for database cursors with automatic commit."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            yield cur


def execute(query: str, params: tuple = None) -> None:
    """Execute a query without returning results."""
    with get_cursor() as cur:
        cur.execute(query, params)


def fetch_one(query: str, params: tuple = None) -> dict | None:
    """Execute a query and return a single row as dict."""
    with get_connection() as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(query, params)
            return cur.fetchone()


def fetch_all(query: str, params: tuple = None) -> list[dict]:
    """Execute a query and return all rows as list of dicts."""
    with get_connection() as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(query, params)
            return cur.fetchall()


def fetch_dataframe(query: str, params: tuple = None):
    """Execute a query and return results as pandas DataFrame."""
    import pandas as pd
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=params)
