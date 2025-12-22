"""Backfill historical data for all active instruments."""

from datetime import date, timedelta

from omnivore.instrument import InstrumentService
from omnivore.services import FeatureEngine


def main():
    instruments = InstrumentService()
    feature_engine = FeatureEngine()

    # Get all active instruments
    instruments = instruments.repository.list_instruments(active_only=True)

    # Backfill 5 years of data
    start_date = date.today() - timedelta(days=5 * 365)

    for instrument in instruments:
        print(f"\nProcessing {instrument['symbol']}...")

        # Fetch OHLCV data
        print("  Fetching OHLCV data...")
        result = instruments.refresh(
            symbol=instrument["symbol"],
            start_date=start_date,
        )
        print(f"  Stored {result['rows_stored']} rows")

        # Compute features
        print("  Computing features...")
        feature_result = feature_engine.compute_and_store(
            instrument_id=instrument["id"],
            start_date=start_date,
        )
        print(f"  Computed features for {feature_result['rows_stored']} rows")

    print("\nBackfill complete!")


if __name__ == "__main__":
    main()
