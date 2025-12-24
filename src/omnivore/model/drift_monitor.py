import json
from datetime import date, timedelta
import numpy as np
import pandas as pd
from omnivore import db
from omnivore.services.feature_engine import FeatureEngine


class DriftMonitor:
    """Monitors feature and prediction drift."""

    def __init__(self):
        self.feature_engine = FeatureEngine()

    def compute_feature_drift(
        self,
        model_version_id: int,
        instrument_id: int,
        reference_start: date,
        reference_end: date,
        current_start: date,
        current_end: date,
    ) -> dict:
        """
        Compute feature drift using Population Stability Index (PSI).
        Compares feature distributions between reference and current periods.
        """
        # Get model version to determine features
        version = db.fetch_one(
            """
            SELECT mv.*, m.feature_config
            FROM model_versions mv
            JOIN models m ON m.id = mv.model_id
            WHERE mv.id = %s
            """,
            (model_version_id,)
        )

        feature_names = version["feature_config"]
        if isinstance(feature_names, str):
            feature_names = json.loads(feature_names)

        # Get reference period features
        ref_features = self.feature_engine.get_features(
            instrument_id=instrument_id,
            start_date=reference_start,
            end_date=reference_end,
            include_targets=False,
        )

        # Get current period features
        curr_features = self.feature_engine.get_features(
            instrument_id=instrument_id,
            start_date=current_start,
            end_date=current_end,
            include_targets=False,
        )

        # Compute PSI for each feature
        drift_metrics = {}
        for feature in feature_names:
            if feature not in ref_features.columns or feature not in curr_features.columns:
                continue

            ref_vals = ref_features[feature].dropna()
            curr_vals = curr_features[feature].dropna()

            if len(ref_vals) < 10 or len(curr_vals) < 10:
                continue

            psi = self._compute_psi(ref_vals, curr_vals)
            drift_metrics[feature] = {
                "psi": float(psi),
                "drift_level": self._classify_psi(psi),
                "ref_mean": float(ref_vals.mean()),
                "curr_mean": float(curr_vals.mean()),
                "ref_std": float(ref_vals.std()),
                "curr_std": float(curr_vals.std()),
            }

        # Overall drift assessment
        psi_values = [m["psi"] for m in drift_metrics.values()]
        overall_psi = np.mean(psi_values) if psi_values else 0

        return {
            "overall_psi": float(overall_psi),
            "overall_drift_level": self._classify_psi(overall_psi),
            "feature_drift": drift_metrics,
            "reference_period": {"start": str(reference_start), "end": str(reference_end)},
            "current_period": {"start": str(current_start), "end": str(current_end)},
        }

    def _compute_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Compute Population Stability Index between two distributions."""
        # Create bins based on reference distribution
        _, bin_edges = np.histogram(reference, bins=bins)

        # Count observations in each bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions (add small epsilon to avoid division by zero)
        epsilon = 1e-10
        ref_props = (ref_counts + epsilon) / (len(reference) + epsilon * bins)
        curr_props = (curr_counts + epsilon) / (len(current) + epsilon * bins)

        # Compute PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))

        return psi

    def _classify_psi(self, psi: float) -> str:
        """Classify PSI value into drift severity level."""
        if psi < 0.1:
            return "none"
        elif psi < 0.2:
            return "slight"
        elif psi < 0.25:
            return "moderate"
        else:
            return "significant"

    def compute_prediction_drift(
        self,
        model_version_id: int,
        window_days: int = 30,
    ) -> dict:
        """
        Analyze prediction accuracy drift over time.
        Compares recent prediction accuracy to historical accuracy.
        """
        # Get predictions with actuals
        predictions = db.fetch_all(
            """
            SELECT
                p.prediction_date,
                p.predicted_value,
                pa.actual_value,
                pa.error,
                pa.direction_correct
            FROM predictions p
            JOIN prediction_actuals pa ON pa.prediction_id = p.id
            WHERE p.model_version_id = %s
            ORDER BY p.prediction_date
            """,
            (model_version_id,)
        )

        if len(predictions) < 20:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 20 predictions with actuals, have {len(predictions)}",
            }

        df = pd.DataFrame(predictions)
        df["prediction_date"] = pd.to_datetime(df["prediction_date"])

        # Split into historical and recent
        cutoff = df["prediction_date"].max() - timedelta(days=window_days)
        historical = df[df["prediction_date"] < cutoff]
        recent = df[df["prediction_date"] >= cutoff]

        if len(historical) < 10 or len(recent) < 5:
            return {
                "status": "insufficient_data",
                "message": "Need more data in both periods",
            }

        # Compute metrics for each period
        hist_metrics = {
            "mae": float(historical["error"].abs().mean()),
            "directional_accuracy": float(historical["direction_correct"].mean()),
            "mean_error": float(historical["error"].mean()),
            "count": len(historical),
        }

        recent_metrics = {
            "mae": float(recent["error"].abs().mean()),
            "directional_accuracy": float(recent["direction_correct"].mean()),
            "mean_error": float(recent["error"].mean()),
            "count": len(recent),
        }

        # Compute drift
        mae_drift = (recent_metrics["mae"] - hist_metrics["mae"]) / hist_metrics["mae"]
        acc_drift = recent_metrics["directional_accuracy"] - hist_metrics["directional_accuracy"]

        return {
            "status": "computed",
            "historical": hist_metrics,
            "recent": recent_metrics,
            "drift": {
                "mae_change_pct": float(mae_drift * 100),
                "directional_accuracy_change": float(acc_drift),
                "is_degrading": mae_drift > 0.2 or acc_drift < -0.1,
            },
            "window_days": window_days,
        }

    def create_report(
        self,
        model_version_id: int,
        instrument_id: int,
        drift_type: str,
        metrics: dict,
    ) -> dict:
        """Store a drift report in the database."""
        is_alert = False

        if drift_type == "feature":
            is_alert = metrics.get("overall_drift_level") in ("moderate", "significant")
        elif drift_type == "prediction":
            is_alert = metrics.get("drift", {}).get("is_degrading", False)

        return db.fetch_one(
            """
            INSERT INTO drift_reports (model_version_id, report_date, drift_type, metrics, is_alert)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
            """,
            (model_version_id, date.today(), drift_type, json.dumps(metrics), is_alert)
        )

    def get_reports(
        self,
        model_version_id: int = None,
        drift_type: str = None,
        alerts_only: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        """Query drift reports."""
        query = "SELECT * FROM drift_reports WHERE 1=1"
        params = []

        if model_version_id:
            query += " AND model_version_id = %s"
            params.append(model_version_id)
        if drift_type:
            query += " AND drift_type = %s"
            params.append(drift_type)
        if alerts_only:
            query += " AND is_alert = true"

        query += " ORDER BY report_date DESC LIMIT %s"
        params.append(limit)

        return db.fetch_all(query, tuple(params))
