from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from omnivore.ohlcv import OhlcvRepository
from omnivore.repositories.instrument_repository import InstrumentRepository
from omnivore.repositories.model_repository import ModelRepository
from omnivore.repositories.prediction_repository import PredictionRepository


class DataService:
    """
    Handles business logic and orchestration, using repositories for data access.
    """

    def __init__(self):
        self.instruments = InstrumentRepository()
        self.predictions = PredictionRepository()
        self.ohlcv = OhlcvRepository()
        self.models = ModelRepository()

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: date,
        end_date: date = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance."""
        end_date = end_date or date.today()
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date + timedelta(days=1))

        if df.empty:
            return df

        # Normalize column names
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Ensure we have the expected columns
        df = df.rename(
            columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )

        # Add adj_close (yfinance history already adjusts, so close = adj_close)
        df["adj_close"] = df["close"]

        # Convert date to date type (not datetime)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]

    def clean_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data."""
        if df.empty:
            return df

        # Remove rows with missing critical data
        df = df.dropna(subset=["open", "high", "low", "close"])

        # Ensure high >= low
        df = df[df["high"] >= df["low"]]

        # Ensure high >= open, close and low <= open, close
        df = df[
            (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
        ]

        # Remove zero or negative prices
        price_cols = ["open", "high", "low", "close", "adj_close"]
        for col in price_cols:
            df = df[df[col] > 0]

        # Remove duplicate dates
        df = df.drop_duplicates(subset=["date"], keep="last")

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        return df

    def refresh_instrument(
        self,
        symbol: str,
        start_date: date = None,
        end_date: date = None,
    ) -> dict:
        """
        Fetch, clean, and store data for an instrument.
        Uses InstrumentRepository and OhlcvRepository.
        """
        instrument = self.instruments.get_by_symbol(symbol)
        if not instrument:
            raise ValueError(f"Instrument {symbol} not found")

        # Default to last 5 years if no start date
        if start_date is None:
            start_date = date.today() - timedelta(days=5 * 365)

        df = self.fetch_ohlcv(symbol, start_date, end_date)
        df = self.clean_ohlcv(df)
        inserted = self.ohlcv.store_ohlcv(instrument["id"], df)

        return {
            "instrument_id": instrument["id"],
            "symbol": symbol,
            "rows_fetched": len(df),
            "rows_stored": inserted,
            "date_range": {
                "start": str(df["date"].min()) if not df.empty else None,
                "end": str(df["date"].max()) if not df.empty else None,
            },
        }
