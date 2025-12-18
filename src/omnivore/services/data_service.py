# ============================================================================
# FILE: src/omnivore/services/data_service.py
# ============================================================================
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from omnivore import db


class DataService:
    """Handles fetching, cleaning, and storing OHLCV data."""

    def get_instrument(self, symbol: str) -> dict | None:
        """Get instrument by symbol."""
        return db.fetch_one(
            "SELECT * FROM instruments WHERE symbol = %s",
            (symbol.upper(),)
        )

    def get_instrument_by_id(self, instrument_id: int) -> dict | None:
        """Get instrument by ID."""
        return db.fetch_one(
            "SELECT * FROM instruments WHERE id = %s",
            (instrument_id,)
        )

    def list_instruments(self, active_only: bool = True) -> list[dict]:
        """List all instruments."""
        if active_only:
            return db.fetch_all(
                "SELECT * FROM instruments WHERE is_active = true ORDER BY symbol"
            )
        return db.fetch_all("SELECT * FROM instruments ORDER BY symbol")

    def create_instrument(
        self,
        symbol: str,
        name: str = None,
        asset_type: str = "stock",
        exchange: str = None,
    ) -> dict:
        """Create a new instrument."""
        return db.fetch_one(
            """
            INSERT INTO instruments (symbol, name, asset_type, exchange)
            VALUES (%s, %s, %s, %s)
            RETURNING *
            """,
            (symbol.upper(), name, asset_type, exchange)
        )

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
        df = df.rename(columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })

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
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) &
            (df["low"] <= df["close"])
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

    def store_ohlcv(self, instrument_id: int, df: pd.DataFrame) -> int:
        """Store OHLCV data in database. Returns count of rows inserted."""
        if df.empty:
            return 0

        inserted = 0
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                for _, row in df.iterrows():
                    try:
                        cur.execute(
                            """
                            INSERT INTO ohlcv_daily
                                (instrument_id, date, open, high, low, close, adj_close, volume)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (instrument_id, date) DO UPDATE SET
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                adj_close = EXCLUDED.adj_close,
                                volume = EXCLUDED.volume,
                                fetched_at = now()
                            """,
                            (
                                instrument_id,
                                row["date"],
                                float(row["open"]),
                                float(row["high"]),
                                float(row["low"]),
                                float(row["close"]),
                                float(row["adj_close"]),
                                int(row["volume"]) if pd.notna(row["volume"]) else None,
                            )
                        )
                        inserted += 1
                    except Exception as e:
                        print(f"Error inserting row for date {row['date']}: {e}")
                conn.commit()

        return inserted

    def refresh_instrument(
        self,
        symbol: str,
        start_date: date = None,
        end_date: date = None,
    ) -> dict:
        """Fetch, clean, and store data for an instrument."""
        instrument = self.get_instrument(symbol)
        if not instrument:
            raise ValueError(f"Instrument {symbol} not found")

        # Default to last 5 years if no start date
        if start_date is None:
            start_date = date.today() - timedelta(days=5 * 365)

        df = self.fetch_ohlcv(symbol, start_date, end_date)
        df = self.clean_ohlcv(df)
        inserted = self.store_ohlcv(instrument["id"], df)

        return {
            "instrument_id": instrument["id"],
            "symbol": symbol,
            "rows_fetched": len(df),
            "rows_stored": inserted,
            "date_range": {
                "start": str(df["date"].min()) if not df.empty else None,
                "end": str(df["date"].max()) if not df.empty else None,
            }
        }

    def get_ohlcv(
        self,
        instrument_id: int,
        start_date: date = None,
        end_date: date = None,
    ) -> pd.DataFrame:
        """Retrieve OHLCV data from database as DataFrame."""
        query = """
            SELECT date, open, high, low, close, adj_close, volume
            FROM ohlcv_daily
            WHERE instrument_id = %s
        """
        params = [instrument_id]

        if start_date:
            query += " AND date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND date <= %s"
            params.append(end_date)

        query += " ORDER BY date"

        return db.fetch_dataframe(query, tuple(params))

    def get_latest_date(self, instrument_id: int) -> date | None:
        """Get the most recent date we have data for."""
        result = db.fetch_one(
            "SELECT MAX(date) as max_date FROM ohlcv_daily WHERE instrument_id = %s",
            (instrument_id,)
        )
        return result["max_date"] if result else None
