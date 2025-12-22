from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from omnivore.ohlcv import OhlcvRepository


class OhlcvService:
    """
    Handles business logic and orchestration, using repositories for data access.
    """

    def __init__(self):
        self.repository = OhlcvRepository()

    def fetch(
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

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
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
