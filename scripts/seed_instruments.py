"""Seed initial instruments into the database."""
from omnivore.instrument.repository import InstrumentRepository

INITIAL_INSTRUMENTS = [
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "asset_type": "etf", "exchange": "NYSE"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "asset_type": "etf", "exchange": "NASDAQ"},
    {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "asset_type": "etf", "exchange": "NYSE"},
    {"symbol": "DIA", "name": "SPDR Dow Jones Industrial Average ETF", "asset_type": "etf", "exchange": "NYSE"},
]


def main():
    instruments_repo = InstrumentRepository()

    for instrument in INITIAL_INSTRUMENTS:
        existing = instruments_repo.get_instrument(instrument["symbol"])
        if existing:
            print(f"Skipping {instrument['symbol']} - already exists")
            continue

        result = instruments_repo.create_instrument(**instrument)
        print(f"Created: {result['symbol']} (id={result['id']})")


if __name__ == "__main__":
    main()
