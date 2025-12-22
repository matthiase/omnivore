from datetime import date, timedelta

from omnivore.instrument import InstrumentRepository
from omnivore.ohlcv import OhlcvService


class InstrumentService:
    def __init__(self):
        self.repository = InstrumentRepository()

    def refresh(
        self,
        symbol: str,
        start_date: date = None,
        end_date: date = None,
    ) -> dict:
        """
        Fetch, clean, and store data for an instrument.
        """
        ohlcv = OhlcvService()
        instrument = self.repository.get_by_symbol(symbol)
        if not instrument:
            raise ValueError(f"Instrument {symbol} not found")

        # Default to last 5 years if no start date
        if start_date is None:
            start_date = date.today() - timedelta(days=5 * 365)

        df = ohlcv.fetch(symbol, start_date, end_date)
        df = ohlcv.clean(df)
        inserted = ohlcv.repository.store_ohlcv(instrument["id"], df)

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
