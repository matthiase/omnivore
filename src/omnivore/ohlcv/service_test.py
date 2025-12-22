"""
Integration tests for OhlcvService.

Run with: OMNIVORE_ENV=test pytest src/omnivore/ohlcv/service_test.py -v
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from omnivore.ohlcv.service import OhlcvService


def make_yfinance_response(rows: list[dict]) -> pd.DataFrame:
    """
    Helper to create DataFrames that mimic yfinance response format.

    yfinance returns DataFrames with:
    - DatetimeIndex named 'Date'
    - Columns: Open, High, Low, Close, Volume (capitalized)
    """
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


def make_ohlcv_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Helper to create cleaned OHLCV DataFrames for testing."""
    return pd.DataFrame(rows)


class TestFetch:
    """Tests for OhlcvService.fetch()"""

    @pytest.mark.parametrize(
        "rows,expected_count",
        [
            # Single row
            (
                [
                    {
                        "Date": "2024-01-02",
                        "Open": 100.0,
                        "High": 101.0,
                        "Low": 99.0,
                        "Close": 100.5,
                        "Volume": 1000000,
                    }
                ],
                1,
            ),
            # Multiple rows
            (
                [
                    {
                        "Date": "2024-01-02",
                        "Open": 100.0,
                        "High": 101.0,
                        "Low": 99.0,
                        "Close": 100.5,
                        "Volume": 1000000,
                    },
                    {
                        "Date": "2024-01-03",
                        "Open": 101.0,
                        "High": 102.0,
                        "Low": 100.0,
                        "Close": 101.5,
                        "Volume": 1100000,
                    },
                    {
                        "Date": "2024-01-04",
                        "Open": 102.0,
                        "High": 103.0,
                        "Low": 101.0,
                        "Close": 102.5,
                        "Volume": 1200000,
                    },
                ],
                3,
            ),
        ],
    )
    def test_fetch_success(self, rows, expected_count):
        service = OhlcvService()
        mock_df = make_yfinance_response(rows)

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_df
            mock_ticker_class.return_value = mock_ticker

            result = service.fetch("SPY", start_date=date(2024, 1, 1), end_date=date(2024, 1, 5))

        assert len(result) == expected_count
        assert list(result.columns) == [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
        mock_ticker_class.assert_called_once_with("SPY")

    def test_fetch_empty_response(self):
        service = OhlcvService()

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_ticker_class.return_value = mock_ticker

            result = service.fetch("INVALID", start_date=date(2024, 1, 1))

        assert result.empty

    def test_fetch_normalizes_column_names(self):
        service = OhlcvService()
        mock_df = make_yfinance_response(
            [
                {
                    "Date": "2024-01-02",
                    "Open": 100.0,
                    "High": 101.0,
                    "Low": 99.0,
                    "Close": 100.5,
                    "Volume": 1000000,
                }
            ]
        )

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_df
            mock_ticker_class.return_value = mock_ticker

            result = service.fetch("SPY", start_date=date(2024, 1, 1))

        # All columns should be lowercase
        assert all(col.islower() for col in result.columns)

    def test_fetch_adds_adj_close(self):
        service = OhlcvService()
        mock_df = make_yfinance_response(
            [
                {
                    "Date": "2024-01-02",
                    "Open": 100.0,
                    "High": 101.0,
                    "Low": 99.0,
                    "Close": 100.5,
                    "Volume": 1000000,
                }
            ]
        )

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_df
            mock_ticker_class.return_value = mock_ticker

            result = service.fetch("SPY", start_date=date(2024, 1, 1))

        assert "adj_close" in result.columns
        assert result.iloc[0]["adj_close"] == result.iloc[0]["close"]

    def test_fetch_converts_date_to_date_type(self):
        service = OhlcvService()
        mock_df = make_yfinance_response(
            [
                {
                    "Date": "2024-01-02",
                    "Open": 100.0,
                    "High": 101.0,
                    "Low": 99.0,
                    "Close": 100.5,
                    "Volume": 1000000,
                }
            ]
        )

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_df
            mock_ticker_class.return_value = mock_ticker

            result = service.fetch("SPY", start_date=date(2024, 1, 1))

        assert isinstance(result.iloc[0]["date"], date)

    def test_fetch_uses_today_as_default_end_date(self):
        service = OhlcvService()

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_ticker_class.return_value = mock_ticker

            service.fetch("SPY", start_date=date(2024, 1, 1))

        # Verify history was called with end_date = today + 1 day
        call_kwargs = mock_ticker.history.call_args
        assert call_kwargs is not None


class TestClean:
    """Tests for OhlcvService.clean() - pure function, no mocking needed"""

    def test_clean_valid_data_unchanged(self):
        service = OhlcvService()
        df = make_ohlcv_dataframe(
            [
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
                    "open": 101.0,
                    "high": 102.0,
                    "low": 100.0,
                    "close": 101.5,
                    "adj_close": 101.5,
                    "volume": 1100000,
                },
            ]
        )

        result = service.clean(df)

        assert len(result) == 2

    def test_clean_empty_dataframe(self):
        service = OhlcvService()
        df = pd.DataFrame()

        result = service.clean(df)

        assert result.empty

    @pytest.mark.parametrize("missing_col", ["open", "high", "low", "close"])
    def test_clean_removes_rows_with_missing_prices(self, missing_col):
        service = OhlcvService()
        df = make_ohlcv_dataframe(
            [
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
                    "open": 101.0,
                    "high": 102.0,
                    "low": 100.0,
                    "close": 101.5,
                    "adj_close": 101.5,
                    "volume": 1100000,
                },
            ]
        )
        df.loc[0, missing_col] = None

        result = service.clean(df)

        assert len(result) == 1
        assert result.iloc[0]["date"] == date(2024, 1, 2)

    @pytest.mark.parametrize(
        "high,low,should_keep",
        [
            (101.0, 99.0, True),  # Valid: high > low
            (100.0, 100.0, True),  # Valid: high == low
            (99.0, 101.0, False),  # Invalid: high < low
        ],
    )
    def test_clean_validates_high_low_relationship(self, high, low, should_keep):
        service = OhlcvService()
        df = make_ohlcv_dataframe(
            [
                {
                    "date": date(2024, 1, 1),
                    "open": 100.0,
                    "high": high,
                    "low": low,
                    "close": 100.0,
                    "adj_close": 100.0,
                    "volume": 1000000,
                },
            ]
        )

        result = service.clean(df)

        assert len(result) == (1 if should_keep else 0)

    @pytest.mark.parametrize(
        "open_,high,close,low,should_keep",
        [
            (100.0, 101.0, 100.5, 99.0, True),  # Valid: high >= open, close; low <= open, close
            (102.0, 101.0, 100.5, 99.0, False),  # Invalid: high < open
            (100.0, 101.0, 102.0, 99.0, False),  # Invalid: high < close
            (100.0, 101.0, 100.5, 100.5, False),  # Invalid: low > open
            (100.0, 101.0, 99.0, 99.5, False),  # Invalid: low > close
        ],
    )
    def test_clean_validates_ohlc_relationships(self, open_, high, close, low, should_keep):
        service = OhlcvService()
        df = make_ohlcv_dataframe(
            [
                {
                    "date": date(2024, 1, 1),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": 1000000,
                },
            ]
        )

        result = service.clean(df)

        assert len(result) == (1 if should_keep else 0)

    @pytest.mark.parametrize("price_col", ["open", "high", "low", "close", "adj_close"])
    def test_clean_removes_zero_prices(self, price_col):
        service = OhlcvService()
        df = make_ohlcv_dataframe(
            [
                {
                    "date": date(2024, 1, 1),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "adj_close": 100.5,
                    "volume": 1000000,
                },
            ]
        )
        df.loc[0, price_col] = 0.0

        result = service.clean(df)

        assert len(result) == 0

    @pytest.mark.parametrize("price_col", ["open", "high", "low", "close", "adj_close"])
    def test_clean_removes_negative_prices(self, price_col):
        service = OhlcvService()
        df = make_ohlcv_dataframe(
            [
                {
                    "date": date(2024, 1, 1),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "adj_close": 100.5,
                    "volume": 1000000,
                },
            ]
        )
        df.loc[0, price_col] = -1.0

        result = service.clean(df)

        assert len(result) == 0

    def test_clean_removes_duplicate_dates_keeps_last(self):
        service = OhlcvService()
        df = make_ohlcv_dataframe(
            [
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
                    "date": date(2024, 1, 1),
                    "open": 200.0,
                    "high": 201.0,
                    "low": 199.0,
                    "close": 200.5,
                    "adj_close": 200.5,
                    "volume": 2000000,
                },
            ]
        )

        result = service.clean(df)

        assert len(result) == 1
        assert result.iloc[0]["open"] == 200.0  # Kept the last one

    def test_clean_sorts_by_date(self):
        service = OhlcvService()
        df = make_ohlcv_dataframe(
            [
                {
                    "date": date(2024, 1, 3),
                    "open": 103.0,
                    "high": 104.0,
                    "low": 102.0,
                    "close": 103.5,
                    "adj_close": 103.5,
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
                    "open": 101.0,
                    "high": 102.0,
                    "low": 100.0,
                    "close": 101.5,
                    "adj_close": 101.5,
                    "volume": 1000000,
                },
            ]
        )

        result = service.clean(df)

        dates = result["date"].tolist()
        assert dates == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]

    def test_clean_resets_index(self):
        service = OhlcvService()
        df = make_ohlcv_dataframe(
            [
                {
                    "date": date(2024, 1, 2),
                    "open": 101.0,
                    "high": 102.0,
                    "low": 100.0,
                    "close": 101.5,
                    "adj_close": 101.5,
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
            ]
        )

        result = service.clean(df)

        assert list(result.index) == [0, 1]
