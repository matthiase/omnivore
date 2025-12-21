"""
Integration tests for InstrumentRepository.

Run with: OMNIVORE_ENV=test pytest src/omnivore/repositories/instrument_repository_test.py -v
"""
import pytest
from psycopg.errors import UniqueViolation

from omnivore.repositories.instrument_repository import InstrumentRepository


class TestCreate:
    """Tests for InstrumentRepository.create()"""

    @pytest.mark.parametrize("symbol,name,asset_type,exchange", [
        ("AAPL", "Apple Inc.", "stock", "NASDAQ"),
        ("SPY", "SPDR S&P 500 ETF", "etf", "NYSE"),
        ("msft", None, "stock", None),  # lowercase, minimal fields
    ])
    def test_create_success(self, db_connection, symbol, name, asset_type, exchange):
        repo = InstrumentRepository()

        result = repo.create(symbol=symbol, name=name, asset_type=asset_type, exchange=exchange)

        assert result["id"] is not None
        assert result["symbol"] == symbol.upper()
        assert result["name"] == name
        assert result["asset_type"] == asset_type
        assert result["exchange"] == exchange
        assert result["is_active"] is True
        assert result["created_at"] is not None

    def test_create_duplicate_symbol_raises(self, db_connection):
        repo = InstrumentRepository()
        repo.create(symbol="DUP", asset_type="stock")

        with pytest.raises(UniqueViolation):
            repo.create(symbol="DUP", asset_type="stock")


class TestGetById:
    """Tests for InstrumentRepository.get_by_id()"""

    @pytest.mark.parametrize("symbol,asset_type", [
        ("AAPL", "stock"),
        ("SPY", "etf"),
    ])
    def test_get_by_id_success(self, db_connection, symbol, asset_type):
        repo = InstrumentRepository()
        created = repo.create(symbol=symbol, asset_type=asset_type)

        result = repo.get_by_id(created["id"])

        assert result is not None
        assert result["id"] == created["id"]
        assert result["symbol"] == symbol.upper()

    def test_get_by_id_not_found(self, db_connection):
        repo = InstrumentRepository()

        result = repo.get_by_id(99999)

        assert result is None


class TestGetBySymbol:
    """Tests for InstrumentRepository.get_by_symbol()"""

    @pytest.mark.parametrize("stored_symbol,lookup_symbol", [
        ("AAPL", "AAPL"),   # exact match
        ("MSFT", "msft"),   # lowercase lookup
        ("GOOG", "gOoG"),   # mixed case lookup
    ])
    def test_get_by_symbol_success(self, db_connection, stored_symbol, lookup_symbol):
        repo = InstrumentRepository()
        repo.create(symbol=stored_symbol, asset_type="stock")

        result = repo.get_by_symbol(lookup_symbol)

        assert result is not None
        assert result["symbol"] == stored_symbol.upper()

    def test_get_by_symbol_not_found(self, db_connection):
        repo = InstrumentRepository()

        result = repo.get_by_symbol("DOESNOTEXIST")

        assert result is None


class TestList:
    """Tests for InstrumentRepository.list()"""

    @pytest.mark.parametrize("active_only,expected_count", [
        (True, 2),   # Only active instruments
        (False, 3),  # All instruments including deactivated
    ])
    def test_list(self, db_connection, active_only, expected_count):
        repo = InstrumentRepository()
        repo.create(symbol="ACTIVE1", asset_type="stock")
        repo.create(symbol="ACTIVE2", asset_type="stock")
        inactive = repo.create(symbol="INACTIVE", asset_type="stock")

        # Deactivate one instrument directly via SQL
        db_connection.execute(
            "UPDATE instruments SET is_active = false WHERE id = %s",
            (inactive["id"],)
        )

        result = repo.list(active_only=active_only)

        assert len(result) == expected_count
        assert all(isinstance(i, dict) for i in result)

    def test_list_empty(self, db_connection):
        repo = InstrumentRepository()

        result = repo.list()

        assert result == []


class TestGetActiveInstruments:
    """Tests for InstrumentRepository.get_active_instruments()"""

    @pytest.mark.parametrize("num_active,num_inactive", [
        (3, 0),  # All active
        (2, 1),  # Some inactive
        (0, 2),  # All inactive
    ])
    def test_get_active_instruments(self, db_connection, num_active, num_inactive):
        repo = InstrumentRepository()

        # Create active instruments
        for i in range(num_active):
            repo.create(symbol=f"ACTIVE{i}", name=f"Active {i}", asset_type="stock")

        # Create and deactivate instruments
        for i in range(num_inactive):
            inactive = repo.create(symbol=f"INACTIVE{i}", asset_type="stock")
            db_connection.execute(
                "UPDATE instruments SET is_active = false WHERE id = %s",
                (inactive["id"],)
            )

        result = repo.get_active_instruments()

        assert len(result) == num_active
        # Verify only expected columns are returned
        if result:
            assert set(result[0].keys()) == {"id", "symbol", "name"}

    def test_get_active_instruments_empty(self, db_connection):
        repo = InstrumentRepository()

        result = repo.get_active_instruments()

        assert result == []
