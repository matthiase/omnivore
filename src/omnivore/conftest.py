# src/omnivore/conftest.py
"""
Pytest configuration and shared fixtures for integration tests.

Tests are co-located with implementation files using the *_test.py suffix.
This file provides fixtures available to all tests in the package.
"""

import os

# Set environment BEFORE importing any app modules
os.environ["OMNIVORE_ENV"] = "test"

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row

from omnivore import db
from omnivore.config import config

# =============================================================================
# Database Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_db():
    """
    Create test database and schema once per test session.

    This fixture:
    1. Drops the test database if it exists (clean slate)
    2. Creates a fresh test database
    3. Applies all migrations

    Runs once at the start of the test session.
    """
    test_db_url = config.database_url

    # Parse connection string to get database name
    # e.g., "postgresql://user:pass@localhost:5432/omnivore_test" -> "omnivore_test"
    base_url = test_db_url.rsplit("/", 1)[0] + "/postgres"
    db_name = test_db_url.rsplit("/", 1)[1].split("?")[0]

    # Connect to postgres database to create/drop test database
    with psycopg.connect(base_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Terminate existing connections to test database
            cur.execute(
                """
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = %s
                AND pid <> pg_backend_pid()
            """,
                (db_name,),
            )

            cur.execute(f"DROP DATABASE IF EXISTS {db_name}")
            cur.execute(f"CREATE DATABASE {db_name}")

    # Apply schema migrations
    migrations_dir = Path(__file__).parent.parent.parent / "migrations"
    schema_file = migrations_dir / "001_initial_schema.sql"

    if not schema_file.exists():
        raise FileNotFoundError(f"Migration file not found: {schema_file}")

    with psycopg.connect(test_db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_file.read_text())
        conn.commit()

    yield test_db_url

    # Optional: uncomment to drop database after test session
    # with psycopg.connect(base_url, autocommit=True) as conn:
    #     with conn.cursor() as cur:
    #         cur.execute(f"DROP DATABASE IF EXISTS {db_name}")


@pytest.fixture
def db_connection(test_db):
    """
    Provide a database connection with transaction rollback.

    Each test runs in a transaction that is rolled back at the end,
    ensuring tests don't affect each other. This is faster than
    truncating tables between tests.
    """
    conn = psycopg.connect(config.database_url)

    # Clean slate: truncate all tables before each test
    with conn.cursor() as cur:
        cur.execute("""
            TRUNCATE instruments, ohlcv_daily, features_daily,
                     models, model_versions, predictions,
                     prediction_actuals, drift_reports
            RESTART IDENTITY CASCADE
        """)
    conn.commit()

    # Override the db module to use this connection
    db.set_connection_override(conn)

    yield conn

    # Rollback any changes made during the test
    conn.rollback()
    db.clear_connection_override()
    conn.close()


@pytest.fixture
def db_cursor(db_connection):
    """Provide a cursor for direct SQL operations in tests."""
    with db_connection.cursor(row_factory=dict_row) as cur:
        yield cur


# =============================================================================
# Repository Fixtures
# =============================================================================


@pytest.fixture
def instrument_repo(db_connection):
    """Provide an InstrumentRepository instance."""
    from omnivore.instrument import InstrumentRepository

    return InstrumentRepository()


@pytest.fixture
def model_repo(db_connection):
    """Provide a ModelRepository instance (when implemented)."""
    # from omnivore.repositories import ModelRepository
    # return ModelRepository()
    pass


# =============================================================================
# Service Fixtures
# =============================================================================


@pytest.fixture
def data_service(db_connection):
    """Provide a DataService instance."""
    from omnivore.services import DataService

    return DataService()


@pytest.fixture
def feature_engine(db_connection):
    """Provide a FeatureEngine instance."""
    from omnivore.services import FeatureEngine

    return FeatureEngine()


@pytest.fixture
def prediction_service(db_connection):
    """Provide a PredictionService instance."""
    from omnivore.services import PredictionService

    return PredictionService()


# =============================================================================
# Seed Data Fixtures
# =============================================================================


@pytest.fixture
def sample_instrument(db_connection) -> dict:
    """Create a single test instrument."""
    from omnivore.instrument import InstrumentRepository

    repo = InstrumentRepository()
    return repo.create(
        symbol="TEST",
        name="Test Instrument",
        asset_type="stock",
        exchange="NYSE",
    )


@pytest.fixture
def sample_instruments(db_connection) -> list[dict]:
    """Create multiple test instruments."""
    from omnivore.instrument import InstrumentRepository

    repo = InstrumentRepository()

    instruments = [
        {"symbol": "TEST1", "name": "Test Stock 1", "asset_type": "stock", "exchange": "NYSE"},
        {"symbol": "TEST2", "name": "Test Stock 2", "asset_type": "stock", "exchange": "NASDAQ"},
        {"symbol": "TESTETF", "name": "Test ETF", "asset_type": "etf", "exchange": "NYSE"},
    ]

    return [repo.create(**i) for i in instruments]


@pytest.fixture
def sample_ohlcv(db_connection, sample_instrument) -> dict:
    """
    Seed OHLCV data for a test instrument.

    Creates 100 trading days of price data with predictable values
    for easy assertion in tests.

    Returns:
        dict with 'instrument' and 'dates' keys
    """
    instrument = sample_instrument
    dates = []

    with db_connection.cursor() as cur:
        base_price = Decimal("100.00")
        current_date = date.today() - timedelta(days=365)

        for i in range(100):
            # Skip weekends
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)

            # Predictable price movement: +0.50 per day
            price = base_price + Decimal(str(i)) * Decimal("0.50")

            cur.execute(
                """
                INSERT INTO ohlcv_daily
                    (instrument_id, date, open, high, low, close, adj_close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    instrument.id,
                    current_date,
                    price,
                    price + Decimal("1.00"),
                    price - Decimal("0.50"),
                    price + Decimal("0.25"),
                    price + Decimal("0.25"),
                    1_000_000 + i * 10_000,
                ),
            )
            dates.append(current_date)
            current_date += timedelta(days=1)

    return {
        "instrument": instrument,
        "dates": dates,
        "start_date": dates[0],
        "end_date": dates[-1],
    }


@pytest.fixture
def sample_features(db_connection, sample_ohlcv) -> dict:
    """
    Seed computed features for a test instrument.

    Depends on sample_ohlcv fixture.
    """
    instrument = sample_ohlcv["instrument"]

    with db_connection.cursor() as cur:
        for i, d in enumerate(sample_ohlcv["dates"]):
            # Create predictable feature values
            cur.execute(
                """
                INSERT INTO features_daily
                    (instrument_id, date, rsi_14, ma_10, ma_20, ma_50, atr_14, return_1d, return_5d)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    instrument.id,
                    d,
                    50.0 + (i % 30),  # RSI oscillates 50-80
                    100.0 + i * 0.5,  # MA tracks price
                    100.0 + i * 0.45,
                    100.0 + i * 0.4,
                    1.5,  # Constant ATR
                    0.005 if i % 2 == 0 else -0.003,  # Alternating returns
                    0.02 if i % 3 == 0 else -0.01,
                ),
            )

    return {
        "instrument": instrument,
        "dates": sample_ohlcv["dates"],
    }


@pytest.fixture
def sample_model(db_connection) -> dict:
    """Create a test model definition."""
    import json

    with db_connection.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO models (name, description, target, model_type, feature_config, hyperparameters)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                "test_model",
                "A model for testing",
                "return_1d",
                "ridge",
                json.dumps(["rsi_14", "ma_10", "ma_20"]),
                json.dumps({"alpha": 1.0}),
            ),
        )
        return cur.fetchone()


# =============================================================================
# Flask App Fixtures
# =============================================================================


@pytest.fixture
def app(db_connection):
    """Create Flask application for testing."""
    from omnivore.api.app import create_app

    app = create_app()
    app.config["TESTING"] = True

    return app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def temp_model_storage(tmp_path):
    """Provide a temporary directory for model artifacts."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Override config model storage path
    from omnivore import config as cfg

    original_path = cfg.config.model_storage_path
    cfg.config = cfg.Config(
        database_url=cfg.config.database_url,
        redis_url=cfg.config.redis_url,
        model_storage_path=models_dir,
        features_config_path=cfg.config.features_config_path,
        env=cfg.config.env,
    )

    yield models_dir

    # Restore original (optional, tests are isolated anyway)
    cfg.config = cfg.Config(
        database_url=cfg.config.database_url,
        redis_url=cfg.config.redis_url,
        model_storage_path=original_path,
        features_config_path=cfg.config.features_config_path,
        env=cfg.config.env,
    )
