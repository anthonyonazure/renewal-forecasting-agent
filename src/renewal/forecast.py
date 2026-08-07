"""Aggregate per-account forecasts into ARR-at-risk by quarter + summary."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any


def _quarter_label(d: date) -> str:
    return f"{d.year}-Q{(d.month - 1) // 3 + 1}"


def aggregate(scored: list[dict]) -> dict[str, Any]:
    if not scored:
        return {
            "total_arr": 0.0,
            "total_at_risk": 0.0,
            "avg_renew_probability": 0.0,
            "by_quarter": [],
            "high_risk_accounts": [],
            "expansion_candidates": [],
        }

    total_arr = sum(s["arr"] for s in scored)
    total_at_risk = sum(s["arr_at_risk"] for s in scored)
    avg_prob = sum(s["renew_probability"] for s in scored) / len(scored)

    by_q: dict[str, dict[str, float]] = defaultdict(
        lambda: {"arr": 0.0, "at_risk": 0.0, "count": 0}
    )
    for s in scored:
        d = (
            s["contract_end"]
            if isinstance(s["contract_end"], date)
            else date.fromisoformat(str(s["contract_end"]))
        )
        q = _quarter_label(d)
        by_q[q]["arr"] += s["arr"]
        by_q[q]["at_risk"] += s["arr_at_risk"]
        by_q[q]["count"] += 1
    # Sort the quarter labels themselves; the rows mix a string with floats, so
    # sorting the assembled dicts loses the key's type.
    by_quarter = [{"quarter": q, **by_q[q]} for q in sorted(by_q)]

    # High risk = renew_prob < 0.5, sorted by ARR at risk
    high_risk = sorted(
        [s for s in scored if s["renew_probability"] < 0.5],
        key=lambda s: s["arr_at_risk"],
        reverse=True,
    )
    # Expansion candidates = renew_prob > 0.85 AND positive usage growth
    expansion = sorted(
        [
            s
            for s in scored
            if s["renew_probability"] > 0.85
            and s["features"].get("usage_growth_pct_div_100", 0) > 0.3
        ],
        key=lambda s: s["arr"],
        reverse=True,
    )

    return {
        "total_arr": round(total_arr, 2),
        "total_at_risk": round(total_at_risk, 2),
        "avg_renew_probability": round(avg_prob, 4),
        "by_quarter": by_quarter,
        "high_risk_accounts": high_risk,
        "expansion_candidates": expansion,
    }
