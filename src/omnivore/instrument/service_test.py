"""
Integration tests for InstrumentService.

Run with: OMNIVORE_ENV=test pytest src/omnivore/instrument/service_test.py -v
"""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from omnivore.instrument.service import InstrumentService


def make_ohlcv_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Helper to create OHLCV DataFrames for testing."""
    return pd.DataFrame(rows)


def make_yfinance_response(rows: list[dict]) -> pd.DataFrame:
    """
    Helper to create DataFrames that mimic yfinance response format.
    """
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


class TestRefresh:
    """Tests for InstrumentService.refresh()"""

    @pytest.mark.parametrize("num_rows", [1, 5, 10])
    def test_refresh_success(self, db_connection, sample_instrument, num_rows):
        service = InstrumentService()
        symbol = sample_instrument["symbol"]

        mock_yf_data = make_yfinance_response([
            {
                "Date": f"2024-01-{i+1:02d}",
                "Open": 100.0 + i,
                "High": 101.0 + i,
                "Low": 99.0 + i,
                "Close": 100.5 + i,
                "Volume": 1000000 + i * 1000,
            }
            for i in range(num_rows)
        ])

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_yf_data
            mock_ticker_class.return_value = mock_ticker

            result = service.refresh(symbol, start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))

        assert result["instrument_id"] == sample_instrument["id"]
        assert result["symbol"] == symbol
        assert result["rows_fetched"] == num_rows
        assert result["rows_stored"] == num_rows

    def test_refresh_returns_date_range(self, db_connection, sample_instrument):
        service = InstrumentService()

        mock_yf_data = make_yfinance_response([
            {"Date": "2024-01-05", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000000},
            {"Date": "2024-01-10", "Open": 101.0, "High": 102.0, "Low": 100.0, "Close": 101.5, "Volume": 1100000},
            {"Date": "2024-01-15", "Open": 102.0, "High": 103.0, "Low": 101.0, "Close": 102.5, "Volume": 1200000},
        ])

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_yf_data
            mock_ticker_class.return_value = mock_ticker

            result = service.refresh(sample_instrument["symbol"], start_date=date(2024, 1, 1))

        assert result["date_range"]["start"] == "2024-01-05"
        assert result["date_range"]["end"] == "2024-01-15"

    def test_refresh_instrument_not_found_raises(self, db_connection):
        service = InstrumentService()

        with pytest.raises(ValueError, match="not found"):
            service.refresh("DOESNOTEXIST", start_date=date(2024, 1, 1))

    def test_refresh_empty_response(self, db_connection, sample_instrument):
        service = InstrumentService()

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_ticker_class.return_value = mock_ticker

            result = service.refresh(sample_instrument["symbol"], start_date=date(2024, 1, 1))

        assert result["rows_fetched"] == 0
        assert result["rows_stored"] == 0
        assert result["date_range"]["start"] is None
        assert result["date_range"]["end"] is None

    def test_refresh_uses_default_start_date(self, db_connection, sample_instrument):
        service = InstrumentService()

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_ticker_class.return_value = mock_ticker

            service.refresh(sample_instrument["symbol"])

            # Verify history was called with start_date ~5 years ago
            call_kwargs = mock_ticker.history.call_args.kwargs
            expected_start = date.today() - timedelta(days=5 * 365)
            assert call_kwargs["start"] == expected_start

    def test_refresh_cleans_data_before_storing(self, db_connection, sample_instrument):
        service = InstrumentService()

        # Include one invalid row (high < low) that should be cleaned out
        mock_yf_data = make_yfinance_response([
            {"Date": "2024-01-01", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000000},
            {"Date": "2024-01-02", "Open": 100.0, "High": 99.0, "Low": 101.0, "Close": 100.5, "Volume": 1000000},  # Invalid
            {"Date": "2024-01-03", "Open": 102.0, "High": 103.0, "Low": 101.0, "Close": 102.5, "Volume": 1200000},
        ])

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_yf_data
            mock_ticker_class.return_value = mock_ticker

            result = service.refresh(sample_instrument["symbol"], start_date=date(2024, 1, 1))

        # Only 2 valid rows should be stored
        assert result["rows_fetched"] == 2
        assert result["rows_stored"] == 2

    def test_refresh_stores_data_in_database(self, db_connection, sample_instrument):
        service = InstrumentService()

        mock_yf_data = make_yfinance_response([
            {"Date": "2024-01-01", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000000},
            {"Date": "2024-01-02", "Open": 101.0, "High": 102.0, "Low": 100.0, "Close": 101.5, "Volume": 1100000},
        ])

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_yf_data
            mock_ticker_class.return_value = mock_ticker

            service.refresh(sample_instrument["symbol"], start_date=date(2024, 1, 1))

        # Verify data was actually stored
        from omnivore.ohlcv import OhlcvRepository
        repo = OhlcvRepository()
        stored = repo.find(sample_instrument["id"])

        assert len(stored) == 2
        assert stored.iloc[0]["date"] == date(2024, 1, 1)
        assert stored.iloc[1]["date"] == date(2024, 1, 2)

    def test_refresh_case_insensitive_symbol_lookup(self, db_connection, sample_instrument):
        service = InstrumentService()

        mock_yf_data = make_yfinance_response([
            {"Date": "2024-01-01", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000000},
        ])

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_yf_data
            mock_ticker_class.return_value = mock_ticker

            # Use lowercase symbol
            result = service.refresh(sample_instrument["symbol"].lower(), start_date=date(2024, 1, 1))

        assert result["instrument_id"] == sample_instrument["id"]
