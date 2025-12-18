"""Seed initial instruments into the database."""
from omnivore.services import DataService

INITIAL_INSTRUMENTS = [
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "asset_type": "etf", "exchange": "NYSE"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "asset_type": "etf", "exchange": "NASDAQ"},
    {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "asset_type": "etf", "exchange": "NYSE"},
    {"symbol": "DIA", "name": "SPDR Dow Jones Industrial Average ETF", "asset_type": "etf", "exchange": "NYSE"},
]


def main():
    data_service = DataService()

    for instrument in INITIAL_INSTRUMENTS:
        existing = data_service.get_instrument(instrument["symbol"])
        if existing:
            print(f"Skipping {instrument['symbol']} - already exists")
            continue

        result = data_service.create_instrument(**instrument)
        print(f"Created: {result['symbol']} (id={result['id']})")


if __name__ == "__main__":
    main()
