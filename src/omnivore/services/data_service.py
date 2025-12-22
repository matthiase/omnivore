from datetime import date, timedelta

from omnivore.instrument import InstrumentRepository
from omnivore.ohlcv import OhlcvRepository
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
