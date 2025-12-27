#!/usr/bin/env python3
"""Omnivore CLI for daily operations."""

import argparse
from datetime import datetime

import questionary
from rich.console import Console

from omnivore.instrument.repository import InstrumentRepository
from omnivore.ohlcv.repository import OhlcvRepository
from omnivore.services.feature_engine import FeatureEngine

console = Console()


def select_instrument() -> dict | None:
    """Prompt the user to select an instrument from the active list."""
    instrument_repo = InstrumentRepository()
    instruments = instrument_repo.get_active_instruments()
    if not instruments:
        console.print("[red]No instruments found.[/]")
        return None
    selected = questionary.select(
        "Select an instrument:",
        choices=[
            questionary.Choice(title=f"{r['symbol']} ({r['name']})", value=r) for r in instruments
        ],
    ).ask()
    return selected


def refresh_data():
    """Refresh daily OHLCV data for a selected instrument."""
    instrument = select_instrument()
    if not instrument:
        return

    from omnivore.instrument.service import InstrumentService

    instrument_service = InstrumentService()

    ohlcv_repo = OhlcvRepository()
    # Preview: determine the latest date in DB for this instrument
    latest = ohlcv_repo.get_latest_date(instrument["id"])
    summary = (
        f"Will refresh OHLCV data for [bold]{instrument['symbol']}[/] "
        f"(from {latest if latest else 'beginning'} to today)."
    )
    console.print(f"[yellow]{summary}[/]")

    if not questionary.confirm("Proceed with these changes?").ask():
        console.print("[dim]Cancelled.[/]")
        return

    result = instrument_service.refresh(
        symbol=instrument["symbol"],
        start_date=latest,
    )
    console.print(f"[green]Refreshed data for {instrument['symbol']}.[/]")
    if result:
        console.print(f"Rows updated: {result}")


def record_actuals():
    """Record actual market values for a selected instrument."""
    instrument = select_instrument()
    if not instrument:
        return

    # Preview: show what will be recorded (could be improved with more detail)
    summary = (
        f"Will record actual market values for [bold]{instrument['symbol']}[/].\n"
        "This will update resolved outcomes for recent predictions."
    )
    console.print(f"[yellow]{summary}[/]")

    if not questionary.confirm("Proceed with these changes?").ask():
        console.print("[dim]Cancelled.[/]")
        return

    from omnivore.prediction.service import PredictionService

    prediction_service = PredictionService()
    result = prediction_service.backfill_actuals(instrument["id"])
    console.print(f"[green]Recorded actuals for {instrument['symbol']}.[/]")
    if result:
        console.print(f"Rows updated: {result}")


def recompute_features():
    """Recompute features for a selected instrument."""
    instrument = select_instrument()
    if not instrument:
        return

    # Retrieve the most recent entry in the features_daily table for the
    # selected instrument.
    from omnivore.feature import Feature

    features = Feature.find(instrument["id"], limit=1)
    start = features[0]["date"] if features else None

    # Preview: show date range to be recomputed
    feature_engine = FeatureEngine()
    ohlcv_repo = OhlcvRepository()
    df = ohlcv_repo.find(instrument["id"], start_date=start)
    if df.empty:
        console.print(f"[red]No OHLCV data found for {instrument['symbol']}.[/]")
        return

    if not start:
        start = str(df["date"].min())
    end = str(df["date"].max())

    summary = f"Will recompute features for [bold]{instrument['symbol']}[/] from {start} to {end}."
    console.print(f"[yellow]{summary}[/]")

    if not questionary.confirm("Proceed with these changes?").ask():
        console.print("[dim]Cancelled.[/]")
        return

    result = feature_engine.compute_and_store(instrument["id"])
    console.print(f"[green]Recomputed features for {instrument['symbol']}.[/]")
    if result:
        console.print(f"Rows processed: {result.get('rows_processed', 0)}")
        console.print(f"Rows stored: {result.get('rows_stored', 0)}")


def generate_predictions():
    """Generate daily predictions for a selected instrument."""
    instrument = select_instrument()
    if not instrument:
        return

    # Preview: show what will be predicted (could be improved with more detail)
    summary = f"Will generate daily predictions for [bold]{instrument['symbol']}[/]."
    console.print(f"[yellow]{summary}[/]")

    if not questionary.confirm("Proceed with these changes?").ask():
        console.print("[dim]Cancelled.[/]")
        return

    from omnivore.prediction import PredictionService

    prediction_service = PredictionService()
    now = datetime.now()
    result = prediction_service.generate_prediction(
        model_id=1,
        instrument_id=instrument["id"],
        prediction_date=now.date(),
        horizon="1d",
    )

    console.print(f"[green]Generated predictions for {instrument['symbol']}.[/]")
    if result:
        console.print(f"Rows updated: {result}")


def main():
    parser = argparse.ArgumentParser(description="Omnivore CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("refresh-data", help="Refresh daily OHLCV data")
    subparsers.add_parser("record-actuals", help="Record actual market values")
    subparsers.add_parser("recompute-features", help="Recompute features")
    subparsers.add_parser("generate-predictions", help="Generate daily predictions")

    args = parser.parse_args()

    if args.command == "refresh-data":
        refresh_data()
    elif args.command == "record-actuals":
        record_actuals()
    elif args.command == "recompute-features":
        recompute_features()
    elif args.command == "generate-predictions":
        generate_predictions()


if __name__ == "__main__":
    main()
