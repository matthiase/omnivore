from typing import List

from omnivore import db


class Feature:
    @classmethod
    def find(cls, instrument_id: int, limit: int = 10) -> List[dict]:
        return db.fetch_all(
            """
            SELECT *
            FROM features_daily
            WHERE instrument_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (instrument_id, limit),
        )
