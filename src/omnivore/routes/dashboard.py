from flask import Blueprint, render_template
from omnivore.services.data_service import DataService

bp = Blueprint("dashboard", __name__, url_prefix="/")

@bp.route("/")
def dashboard():
    ds = DataService()
    instruments = ds.instruments.get_active_instruments()
    predictions = ds.predictions.get_latest_predictions(horizon="1d")
    accuracy = ds.predictions.get_accuracy_summary(horizon="1d")
    accuracy_map = {row["instrument_id"]: row for row in accuracy}

    return render_template(
        "dashboard.html",
        predictions=predictions,
        accuracy_map=accuracy_map,
        instruments=instruments,
    )
