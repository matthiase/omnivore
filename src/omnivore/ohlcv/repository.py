from datetime import date
from typing import Optional

import pandas as pd

from omnivore import db


class OhlcvRepository:
    """
    Repository for OHLCV (Open, High, Low, Close, Volume) data access.
    Encapsulates all SQL and queries for the ohlcv_daily table.
    """

    def find(
        self,
        instrument_id: int,
        start_date: date = None,
        end_date: date = None,
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data from the database as a pandas DataFrame.
        """
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

    def get_latest_date(self, instrument_id: int) -> Optional[date]:
        """
        Get the most recent date for which we have OHLCV data for a given instrument.
        """
        result = db.fetch_one(
            "SELECT MAX(date) as max_date FROM ohlcv_daily WHERE instrument_id = %s",
            (instrument_id,),
        )
        return result["max_date"] if result else None

    def store_ohlcv(self, instrument_id: int, df: pd.DataFrame) -> int:
        """
        Store OHLCV data in the database. Returns the count of rows inserted or updated.
        """
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
                            ),
                        )
                        inserted += 1
                    except Exception as e:
                        print(f"Error inserting row for date {row['date']}: {e}")
                conn.commit()

        return inserted
