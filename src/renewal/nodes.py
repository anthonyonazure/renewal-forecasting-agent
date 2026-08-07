"""LangGraph nodes for the forecasting pipeline."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import yaml

from renewal.forecast import aggregate
from renewal.model import load_coefficients, score_account
from renewal.state import RState

log = structlog.get_logger()


def _event(kind: str, **detail: Any) -> dict[str, Any]:
    return {"at": datetime.now(UTC).isoformat(), "kind": kind, **detail}


async def load_inputs(state: RState) -> dict[str, Any]:
    accounts = yaml.safe_load(Path(state["accounts_path"]).read_text())["accounts"]
    coefs = load_coefficients(state.get("coefficients_path"))
    log.info("renewal.loaded", accounts=len(accounts), model=coefs["version"])
    return {
        "accounts": accounts,
        "coefficients": coefs,
        "events": [
            _event("inputs_loaded", accounts=len(accounts), model=coefs["version"])
        ],
    }


async def score_all(state: RState) -> dict[str, Any]:
    coefs = state["coefficients"]
    scored = [score_account(a, coefs) for a in state["accounts"]]
    log.info("renewal.scored", n=len(scored))
    return {"scored": scored, "events": [_event("scored", n=len(scored))]}


async def summarize(state: RState) -> dict[str, Any]:
    agg = aggregate(state["scored"])
    log.info(
        "renewal.aggregated",
        total_arr=agg["total_arr"],
        at_risk=agg["total_at_risk"],
        avg_prob=agg["avg_renew_probability"],
    )
    return {
        "aggregate": agg,
        "events": [_event("aggregated", at_risk=agg["total_at_risk"])],
    }


async def build_pdf(state: RState) -> dict[str, Any]:
    """Render a forecast PDF (Jinja2 + WeasyPrint)."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    from weasyprint import HTML

    templates = Path(__file__).resolve().parents[2] / "templates"
    env = Environment(
        loader=FileSystemLoader(templates),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template("forecast.html")
    html = tpl.render(
        run_id=state["run_id"],
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        model_version=state["coefficients"]["version"],
        scored=state["scored"],
        aggregate=state["aggregate"],
    )
    pdf_bytes = HTML(string=html, base_url=str(templates)).write_pdf()
    out_dir = Path(os.environ.get("RENEWAL_OUT_DIR", "forecasts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{state['run_id']}-forecast.pdf"
    pdf_path.write_bytes(pdf_bytes)
    log.info("renewal.pdf.built", path=str(pdf_path), bytes=len(pdf_bytes))
    return {
        "pdf_path": str(pdf_path),
        "events": [_event("pdf_built", path=str(pdf_path))],
    }
