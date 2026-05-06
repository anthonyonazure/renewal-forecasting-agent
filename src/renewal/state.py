"""LangGraph state for the forecasting run."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class RState(TypedDict, total=False):
    run_id: str
    accounts_path: str
    coefficients_path: str

    accounts: list[dict[str, Any]]
    coefficients: dict[str, Any]

    scored: list[dict[str, Any]]
    aggregate: dict[str, Any]

    pdf_path: str | None

    events: Annotated[list[dict[str, Any]], add]
