# ============================================================================
# FILE: src/omnivore/repositories/instrument_repository.py
# ============================================================================
from typing import Optional, List
from omnivore import db


class InstrumentRepository:
    """
    Repository for instrument-related data access.
    Encapsulates all SQL and queries for the instruments table.
    """

    def get_by_symbol(self, symbol: str) -> Optional[dict]:
        """Get instrument by symbol."""
        return db.fetch_one(
            "SELECT * FROM instruments WHERE symbol = %s",
            (symbol.upper(),)
        )

    def get_by_id(self, instrument_id: int) -> Optional[dict]:
        """Get instrument by ID."""
        return db.fetch_one(
            "SELECT * FROM instruments WHERE id = %s",
            (instrument_id,)
        )

    def list(self, active_only: bool = True) -> List[dict]:
        """List all instruments, optionally filtering by active status."""
        if active_only:
            return db.fetch_all(
                "SELECT * FROM instruments WHERE is_active = true ORDER BY symbol"
            )
        return db.fetch_all("SELECT * FROM instruments ORDER BY symbol")

    def create(
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

    def get_active_instruments(self) -> List[dict]:
        """Get all active instruments (id, symbol, name)."""
        return db.fetch_all(
            "SELECT id, symbol, name FROM instruments WHERE is_active = true ORDER BY symbol"
        )