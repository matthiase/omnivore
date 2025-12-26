#!/usr/bin/env python3
"""CLI tool for querying and acting on database records."""

import questionary
from rich.console import Console
from omnivore.instrument.repository import InstrumentRepository

console = Console()


def build_choices(records: list[dict]) -> list[questionary.Choice]:
    """Convert records to questionary choices with the record as the value."""
    return [
        questionary.Choice(
            title=f"{r['symbol']}",
            value=r,
        )
        for r in records
    ]


def handle_selection(record: dict) -> None:
    """Execute logic based on the selected record."""
    console.print(f"\n[bold green]Selected:[/] {record['symbol']}")

    ## Branch based on record properties, or dispatch to other handlers
    # match record.get("status"):
    #    case "pending":
    #        process_pending(record)
    #    case "active":
    #        process_active(record)
    #    case _:
    #        console.print("[yellow]No action defined for this status.[/]")


def process_pending(record: dict) -> None:
    console.print(f"Processing pending record: {record['id']}")
    # Your logic here


def process_active(record: dict) -> None:
    console.print(f"Processing active record: {record['id']}")
    # Your logic here


def main() -> None:
    instrument_repo = InstrumentRepository()

    instruments = instrument_repo.list()

    if not instruments:
        console.print("[red]No instruments found.[/]")
        return

    selected = questionary.select(
        "Select an instrument:",
        choices=[
            questionary.Choice(
                title=r["symbol"],
                value=r,
            )
            for r in instruments
        ],
    ).ask()

    # User pressed Ctrl+C or Escape
    if selected is None:
        console.print("[dim]Cancelled.[/]")
        return

    handle_selection(selected)


if __name__ == "__main__":
    main()
