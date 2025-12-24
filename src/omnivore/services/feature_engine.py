# ============================================================================
# FILE: src/omnivore/services/feature_engine.py
# ============================================================================
from datetime import date

import numpy as np
import pandas as pd
import pandas_ta as ta

from omnivore import db
from omnivore.config import config
from omnivore.instrument.repository import InstrumentRepository
from omnivore.ohlcv.repository import OhlcvRepository
from omnivore.prediction.repository import PredictionRepository
from omnivore.model.repository import ModelRepository


class FeatureEngine:
    """Computes technical indicators and target variables."""

    def __init__(self):
        self.instruments = InstrumentRepository()
        self.predictions = PredictionRepository()
        self.ohlcv = OhlcvRepository()
        self.models = ModelRepository()
        self.config = config.load_features_config()

        # Registry of indicator computation functions
        self.indicator_registry = {
            "RSI": self._compute_rsi,
            "MA": self._compute_ma,
            "ATR": self._compute_atr,
            "MA_CROSS": self._compute_ma_cross,
            "THRESHOLD": self._compute_threshold,
        }

        # Registry of target computation functions
        self.target_registry = {
            "future_return": self._compute_future_return,
        }

    def _compute_rsi(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Compute Relative Strength Index."""
        period = params.get("period", 14)
        return ta.rsi(df["close"], length=period)

    def _compute_ma(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Compute Simple Moving Average."""
        period = params.get("period", 20)
        return ta.sma(df["close"], length=period)

    def _compute_atr(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Compute Average True Range."""
        period = params.get("period", 14)
        return ta.atr(df["high"], df["low"], df["close"], length=period)

    def _compute_ma_cross(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Compute MA crossover signal (fast MA - slow MA)."""
        fast = params.get("fast", 10)
        slow = params.get("slow", 50)
        fast_ma = ta.sma(df["close"], length=fast)
        slow_ma = ta.sma(df["close"], length=slow)
        return fast_ma - slow_ma

    def _compute_threshold(self, df: pd.DataFrame, params: dict, computed: dict) -> pd.Series:
        """Compute threshold-based boolean signal."""
        source = params.get("source")
        if source not in computed:
            raise ValueError(f"Source feature {source} not yet computed")

        series = computed[source]

        if "below" in params:
            return series < params["below"]
        elif "above" in params:
            return series > params["above"]
        else:
            raise ValueError("THRESHOLD requires 'below' or 'above' param")

    def _compute_future_return(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Compute future return (target variable for prediction)."""
        days = params.get("days", 1)
        future_close = df["adj_close"].shift(-days)
        return (future_close - df["adj_close"]) / df["adj_close"]

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all configured features for a DataFrame."""
        result = df.copy()
        computed = {}

        # Compute technical indicators
        for feature_def in self.config["features"]:
            name = feature_def["name"]
            indicator_type = feature_def["type"]
            params = feature_def.get("params", {})

            compute_fn = self.indicator_registry.get(indicator_type)
            if not compute_fn:
                print(f"Warning: Unknown indicator type {indicator_type}")
                continue

            try:
                if indicator_type == "THRESHOLD":
                    result[name] = compute_fn(df, params, computed)
                else:
                    result[name] = compute_fn(df, params)
                computed[name] = result[name]
            except Exception as e:
                print(f"Error computing {name}: {e}")
                result[name] = np.nan

        # Compute target variables
        for target_def in self.config["targets"]:
            name = target_def["name"]
            target_type = target_def["type"]
            params = target_def.get("params", {})

            compute_fn = self.target_registry.get(target_type)
            if compute_fn:
                try:
                    result[name] = compute_fn(df, params)
                except Exception as e:
                    print(f"Error computing target {name}: {e}")
                    result[name] = np.nan

        return result

    def store_features(self, instrument_id: int, df: pd.DataFrame) -> int:
        """Store computed features in database."""
        if df.empty:
            return 0

        inserted = 0
        feature_cols = [f["name"] for f in self.config["features"]]
        target_cols = [t["name"] for t in self.config["targets"]]

        with db.get_connection() as conn:
            with conn.cursor() as cur:
                for _, row in df.iterrows():
                    try:
                        # Build dynamic column list based on what's available
                        cols = ["instrument_id", "date"]
                        vals = [instrument_id, row["date"]]
                        placeholders = ["%s", "%s"]

                        for col in feature_cols + target_cols:
                            if col in row and pd.notna(row[col]):
                                cols.append(col)
                                # Convert numpy/pandas types to Python native
                                val = row[col]
                                if isinstance(val, (np.bool_, bool)):
                                    vals.append(bool(val))
                                elif isinstance(val, (np.floating, float)):
                                    vals.append(float(val))
                                else:
                                    vals.append(val)
                                placeholders.append("%s")

                        # Build upsert query
                        col_str = ", ".join(cols)
                        placeholder_str = ", ".join(placeholders)
                        update_str = ", ".join(
                            f"{c} = EXCLUDED.{c}"
                            for c in cols[2:]  # Skip id and date
                        )

                        query = f"""
                            INSERT INTO features_daily ({col_str})
                            VALUES ({placeholder_str})
                            ON CONFLICT (instrument_id, date) DO UPDATE SET
                                {update_str},
                                computed_at = now()
                        """

                        cur.execute(query, tuple(vals))
                        inserted += 1
                    except Exception as e:
                        print(f"Error storing features for {row['date']}: {e}")

                conn.commit()

        return inserted

    def compute_and_store(
        self,
        instrument_id: int,
        start_date: date = None,
        end_date: date = None,
    ) -> dict:
        """Compute and store features for an instrument."""
        # Get OHLCV data
        df = self.ohlcv.find(instrument_id, start_date, end_date)

        if df.empty:
            return {
                "instrument_id": instrument_id,
                "rows_processed": 0,
                "rows_stored": 0,
            }

        # Compute features
        df_with_features = self.compute_features(df)

        # Store features
        stored = self.store_features(instrument_id, df_with_features)

        return {
            "instrument_id": instrument_id,
            "rows_processed": len(df),
            "rows_stored": stored,
            "date_range": {
                "start": str(df["date"].min()),
                "end": str(df["date"].max()),
            },
        }

    def get_features(
        self,
        instrument_id: int,
        start_date: date = None,
        end_date: date = None,
        include_targets: bool = True,
    ) -> pd.DataFrame:
        """Retrieve computed features from database."""
        # Determine which columns to select
        feature_cols = [f["name"] for f in self.config["features"]]
        target_cols = [t["name"] for t in self.config["targets"]] if include_targets else []

        all_cols = ["date"] + feature_cols + target_cols
        col_str = ", ".join(all_cols)

        query = f"""
            SELECT {col_str}
            FROM features_daily
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

    def get_training_data(
        self,
        instrument_id: int,
        target: str,
        start_date: date = None,
        end_date: date = None,
        feature_names: list[str] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Get feature matrix X and target vector y for training."""
        df = self.get_features(instrument_id, start_date, end_date, include_targets=True)

        if df.empty:
            return pd.DataFrame(), pd.Series()

        # Use specified features or all configured features
        if feature_names is None:
            feature_names = [f["name"] for f in self.config["features"]]

        # Drop rows with NaN in features or target
        df_clean = df.dropna(subset=feature_names + [target])

        X = df_clean[feature_names]
        y = df_clean[target]

        return X, y
