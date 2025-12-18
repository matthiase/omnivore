from flask import Blueprint, render_template
from omnivore.services.data_service import DataService

bp = Blueprint("dashboard", __name__, url_prefix="/")

@bp.route("/")
def dashboard():
    """
    Dashboard page showing the latest prediction for each instrument,
    along with summary statistics.
    """
    ds = DataService()
    instruments = ds.get_active_instruments()
    predictions = ds.get_latest_predictions(horizon="1d")
    accuracy = ds.get_accuracy_summary(horizon="1d")
    accuracy_map = {row["instrument_id"]: row for row in accuracy}

    return render_template(
        "dashboard.html",
        predictions=predictions,
        accuracy_map=accuracy_map,
        instruments=instruments,
    )
