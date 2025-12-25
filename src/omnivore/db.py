"""
Database connection and query utilities.

Provides a simple interface for executing queries with psycopg,
returning results as dictionaries or pandas DataFrames.

For testing, use set_connection_override() to inject a connection
that will be used instead of creating new ones. This enables
transaction rollback between tests.
"""

import decimal
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row

from omnivore.config import config

# =============================================================================
# Connection Override (for testing)
# =============================================================================

_connection_override: psycopg.Connection | None = None


def set_connection_override(conn: psycopg.Connection) -> None:
    """
    Set a connection to use instead of creating new ones.

    Used by test fixtures to ensure all database operations run
    within a single transaction that can be rolled back.

    Args:
        conn: The connection to use for all subsequent operations
    """
    global _connection_override
    _connection_override = conn


def clear_connection_override() -> None:
    """Clear the connection override, restoring normal behavior."""
    global _connection_override
    _connection_override = None


# =============================================================================
# Connection Management
# =============================================================================


@contextmanager
def get_connection():
    """
    Context manager for database connections.

    In normal operation:
        - Opens a new connection
        - Commits on successful exit
        - Rolls back on exception
        - Closes connection when done

    With override set (testing):
        - Returns the override connection
        - Does NOT commit, rollback, or close
        - Caller (test fixture) manages the transaction

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")
    """
    if _connection_override is not None:
        yield _connection_override
        return

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
    """
    Context manager for a cursor with dict rows.

    Convenience wrapper when you just need a cursor.

    Usage:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM instruments")
            rows = cur.fetchall()  # List of dicts
    """
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            yield cur


# =============================================================================
# Query Helpers
# =============================================================================


def execute(query: str, params: tuple = None) -> None:
    """
    Execute a query without returning results.

    Use for INSERT, UPDATE, DELETE when you don't need the affected rows.

    Args:
        query: SQL query with %s placeholders
        params: Tuple of parameter values
    """
    with get_cursor() as cur:
        cur.execute(query, params)


def fetch_one(query: str, params: tuple = None) -> dict[str, Any] | None:
    """
    Execute a query and return a single row as dict.

    Args:
        query: SQL query with %s placeholders
        params: Tuple of parameter values

    Returns:
        Dict of column names to values, or None if no row found
    """
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            return cur.fetchone()


def fetch_all(query: str, params: tuple = None) -> list[dict[str, Any]]:
    """
    Execute a query and return all rows as list of dicts.

    Args:
        query: SQL query with %s placeholders
        params: Tuple of parameter values

    Returns:
        List of dicts, empty list if no rows found
    """
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            return cur.fetchall()


def fetch_dataframe(query: str, params: tuple = None):
    """
    Execute a query and return results as pandas DataFrame.

    Useful for feature engineering and model training where
    pandas operations are needed.

    Args:
        query: SQL query with %s placeholders
        params: Tuple of parameter values

    Returns:
        pandas.DataFrame with query results
    """
    import pandas as pd
    import decimal

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description] if cur.description else []
            df = pd.DataFrame(rows, columns=columns)
            # Convert decimal.Decimal columns to float for numeric compatibility
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, decimal.Decimal)).all():
                    df[col] = df[col].astype(float)
            return df


# =============================================================================
# Batch Operations
# =============================================================================


def execute_many(query: str, params_list: list[tuple]) -> int:
    """
    Execute a query multiple times with different parameters.

    More efficient than calling execute() in a loop for bulk inserts.

    Args:
        query: SQL query with %s placeholders
        params_list: List of parameter tuples

    Returns:
        Number of rows affected
    """
    with get_cursor() as cur:
        cur.executemany(query, params_list)
        return cur.rowcount


def execute_batch(query: str, params_list: list[tuple], page_size: int = 100) -> int:
    """
    Execute a query in batches for very large inserts.

    Uses psycopg's execute_batch for better performance on large datasets.

    Args:
        query: SQL query with %s placeholders
        params_list: List of parameter tuples
        page_size: Number of rows per batch

    Returns:
        Number of parameter sets processed
    """
    with get_cursor() as cur:
        # psycopg3 doesn't have execute_batch like psycopg2
        # Fall back to executemany with chunking
        total = 0
        for i in range(0, len(params_list), page_size):
            chunk = params_list[i : i + page_size]
            cur.executemany(query, chunk)
            total += len(chunk)
        return total
