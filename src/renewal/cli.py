"""Typer CLI."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import structlog
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from renewal.graph import build_graph
from renewal.state import RState

load_dotenv()

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def run(
    accounts: str = typer.Option(None, "--accounts", "-a", help="YAML accounts file"),
    coefficients: str = typer.Option(None, "--coefficients", "-c", help="model coefficients JSON"),
):
    """Score all accounts and emit a forecast PDF."""
    asyncio.run(_run(accounts, coefficients))


async def _run(accounts: str | None, coefs: str | None) -> None:
    run_id = uuid.uuid4().hex[:10]
    initial: RState = {
        "run_id": run_id,
        "accounts_path": accounts or os.environ.get("RENEWAL_ACCOUNTS", "accounts/sample.yaml"),
        "coefficients_path": coefs or os.environ.get("RENEWAL_MODEL_COEFFS", "model/coefficients.json"),
        "events": [],
    }
    graph = build_graph().compile()
    console.rule(f"[bold cyan]Renewal forecast {run_id}[/]")
    final: RState = {}
    async for ev in graph.astream(initial, stream_mode="values"):
        final = ev
        last = (ev.get("events") or [{}])[-1]
        if last:
            console.print(f"  [green]✓[/] {last.get('kind', '?')}")

    agg = final.get("aggregate") or {}
    console.rule("[bold cyan]Aggregate[/]")
    console.print(
        f"Total ARR     : ${agg.get('total_arr', 0):,.0f}\n"
        f"At risk       : ${agg.get('total_at_risk', 0):,.0f}\n"
        f"Avg renew prob: {agg.get('avg_renew_probability', 0):.2%}"
    )

    table = Table(show_header=True, box=None, title="Per-account")
    table.add_column("Account", min_width=24)
    table.add_column("Tier")
    table.add_column("ARR", justify="right")
    table.add_column("Renew prob", justify="right")
    table.add_column("At risk", justify="right")
    for s in sorted(final.get("scored", []), key=lambda s: s["renew_probability"]):
        table.add_row(
            s["account_name"], str(s["tier"]),
            f"${s['arr']:,.0f}",
            f"{s['renew_probability']*100:.0f}%",
            f"${s['arr_at_risk']:,.0f}",
        )
    console.print(table)

    console.print(f"\n[dim]PDF: {final.get('pdf_path')}[/]")
    out = Path("out") / f"{run_id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(final, default=str, indent=2))


@app.command()
def version():
    from renewal import __version__
    console.print(f"renewal-forecasting-agent {__version__}")


if __name__ == "__main__":
    app()
