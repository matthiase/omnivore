from flask import Blueprint, render_template
from omnivore.instrument.repository import InstrumentRepository
from omnivore.prediction.repository import PredictionRepository

bp = Blueprint("dashboard", __name__, url_prefix="/")

@bp.route("/")
def dashboard():
    instruments_repo = InstrumentRepository()
    predictions_repo = PredictionRepository()
    instruments = instruments_repo.get_active_instruments()
    predictions = predictions_repo.get_latest_predictions(horizon="1d")
    accuracy = predictions_repo.get_accuracy_summary(horizon="1d")
    accuracy_map = {row["instrument_id"]: row for row in accuracy}

    # Precompute rows for the template
    instrument_rows = []
    pred_map = {p["instrument_id"]: p for p in predictions}
    for instrument in instruments:
        pred = pred_map.get(instrument["id"])
        acc = accuracy_map.get(instrument["id"])
        instrument_rows.append({
            "instrument": instrument,
            "prediction": pred,
            "accuracy": acc,
        })

    return render_template(
        "dashboard.html",
        instrument_rows=instrument_rows,
    )
