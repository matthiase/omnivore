"""
Integration tests for OhlcvRepository.

Run with: OMNIVORE_ENV=test pytest src/omnivore/ohlcv/repository_test.py -v
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from omnivore.ohlcv import OhlcvRepository


def make_ohlcv_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Helper to create OHLCV DataFrames for testing."""
    return pd.DataFrame(rows)


class TestStoreOhlcv:
    """Tests for OhlcvRepository.store_ohlcv()"""

    @pytest.mark.parametrize("num_rows", [1, 5, 10])
    def test_store_ohlcv_success(self, db_connection, sample_instrument, num_rows):
        repo = OhlcvRepository()
        rows = [
            {
                "date": date(2024, 1, i + 1),
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "adj_close": 100.5 + i,
                "volume": 1000000 + i * 1000,
            }
            for i in range(num_rows)
        ]
        df = make_ohlcv_dataframe(rows)

        result = repo.store_ohlcv(sample_instrument["id"], df)

        assert result == num_rows

    def test_store_ohlcv_empty_dataframe(self, db_connection, sample_instrument):
        repo = OhlcvRepository()
        df = pd.DataFrame()

        result = repo.store_ohlcv(sample_instrument["id"], df)

        assert result == 0

    def test_store_ohlcv_upsert_on_conflict(self, db_connection, sample_instrument):
        repo = OhlcvRepository()
        instrument_id = sample_instrument["id"]

        # Insert initial data
        initial_df = make_ohlcv_dataframe(
            [
                {
                    "date": date(2024, 1, 1),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "adj_close": 100.5,
                    "volume": 1000000,
                }
            ]
        )
        repo.store_ohlcv(instrument_id, initial_df)

        # Update with new values for same date
        updated_df = make_ohlcv_dataframe(
            [
                {
                    "date": date(2024, 1, 1),
                    "open": 200.0,
                    "high": 201.0,
                    "low": 199.0,
                    "close": 200.5,
                    "adj_close": 200.5,
                    "volume": 2000000,
                }
            ]
        )
        result = repo.store_ohlcv(instrument_id, updated_df)

        # Verify update occurred
        stored = repo.find(instrument_id)
        assert result == 1
        assert len(stored) == 1
        assert float(stored.iloc[0]["open"]) == 200.0
        assert float(stored.iloc[0]["close"]) == 200.5

    def test_store_ohlcv_null_volume(self, db_connection, sample_instrument):
        repo = OhlcvRepository()

        df = make_ohlcv_dataframe(
            [
                {
                    "date": date(2024, 1, 1),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "adj_close": 100.5,
                    "volume": None,
                }
            ]
        )

        result = repo.store_ohlcv(sample_instrument["id"], df)

        assert result == 1
        stored = repo.find(sample_instrument["id"])
        assert stored.iloc[0]["volume"] is None


class TestFind:
    """Tests for OhlcvRepository.find()"""

    @pytest.mark.parametrize(
        "start_date,end_date,expected_count",
        [
            (None, None, 5),  # No filter
            (date(2024, 1, 2), None, 4),  # Start filter only
            (None, date(2024, 1, 3), 3),  # End filter only
            (date(2024, 1, 2), date(2024, 1, 4), 3),  # Both filters
            (date(2024, 1, 5), date(2024, 1, 5), 1),  # Single day
        ],
    )
    def test_find_with_date_filters(
        self, db_connection, sample_instrument, start_date, end_date, expected_count
    ):
        repo = OhlcvRepository()
        instrument_id = sample_instrument["id"]

        # Seed 5 days of data
        rows = [
            {
                "date": date(2024, 1, i + 1),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "adj_close": 100.5,
                "volume": 1000000,
            }
            for i in range(5)
        ]
        repo.store_ohlcv(instrument_id, make_ohlcv_dataframe(rows))

        result = repo.find(instrument_id, start_date=start_date, end_date=end_date)

        assert len(result) == expected_count
        assert isinstance(result, pd.DataFrame)

    def test_find_returns_ordered_by_date(self, db_connection, sample_instrument):
        repo = OhlcvRepository()
        instrument_id = sample_instrument["id"]

        # Insert out of order
        rows = [
            {
                "date": date(2024, 1, 3),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "adj_close": 100.5,
                "volume": 1000000,
            },
            {
                "date": date(2024, 1, 1),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "adj_close": 100.5,
                "volume": 1000000,
            },
            {
                "date": date(2024, 1, 2),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "adj_close": 100.5,
                "volume": 1000000,
            },
        ]
        repo.store_ohlcv(instrument_id, make_ohlcv_dataframe(rows))

        result = repo.find(instrument_id)

        dates = result["date"].tolist()
        assert dates == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]

    def test_find_empty_result(self, db_connection, sample_instrument):
        repo = OhlcvRepository()

        result = repo.find(sample_instrument["id"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_find_returns_correct_columns(self, db_connection, sample_instrument):
        repo = OhlcvRepository()
        instrument_id = sample_instrument["id"]

        rows = [
            {
                "date": date(2024, 1, 1),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "adj_close": 100.5,
                "volume": 1000000,
            }
        ]
        repo.store_ohlcv(instrument_id, make_ohlcv_dataframe(rows))

        result = repo.find(instrument_id)

        expected_columns = {"date", "open", "high", "low", "close", "adj_close", "volume"}
        assert set(result.columns) == expected_columns


class TestGetLatestDate:
    """Tests for OhlcvRepository.get_latest_date()"""

    @pytest.mark.parametrize(
        "dates,expected_latest",
        [
            ([date(2024, 1, 1)], date(2024, 1, 1)),
            ([date(2024, 1, 1), date(2024, 1, 15), date(2024, 1, 10)], date(2024, 1, 15)),
            ([date(2024, 12, 31), date(2024, 1, 1)], date(2024, 12, 31)),
        ],
    )
    def test_get_latest_date_success(
        self, db_connection, sample_instrument, dates, expected_latest
    ):
        repo = OhlcvRepository()
        instrument_id = sample_instrument["id"]

        rows = [
            {
                "date": d,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "adj_close": 100.5,
                "volume": 1000000,
            }
            for d in dates
        ]
        repo.store_ohlcv(instrument_id, make_ohlcv_dataframe(rows))

        result = repo.get_latest_date(instrument_id)

        assert result == expected_latest

    def test_get_latest_date_no_data(self, db_connection, sample_instrument):
        repo = OhlcvRepository()

        result = repo.get_latest_date(sample_instrument["id"])

        assert result is None

    def test_get_latest_date_nonexistent_instrument(self, db_connection):
        repo = OhlcvRepository()

        result = repo.get_latest_date(99999)

        assert result is None
