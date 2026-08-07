"""End-to-end: load sample accounts, score, aggregate, render PDF."""

from __future__ import annotations

from pathlib import Path

import pytest

from renewal.graph import build_graph
from renewal.state import RState

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
async def test_full_run(tmp_path, monkeypatch):
    monkeypatch.setenv("RENEWAL_OUT_DIR", str(tmp_path))
    g = build_graph().compile()
    initial: RState = {
        "run_id": "test-run",
        "accounts_path": str(REPO / "accounts" / "sample.yaml"),
        "coefficients_path": str(REPO / "model" / "coefficients.json"),
        "events": [],
    }
    final = await g.ainvoke(initial)

    # 12 accounts in sample
    assert len(final["scored"]) == 12

    # Aggregate has the expected shape
    agg = final["aggregate"]
    assert agg["total_arr"] > 0
    assert 0 <= agg["avg_renew_probability"] <= 1
    assert 0 <= agg["total_at_risk"] <= agg["total_arr"]

    # By-quarter rolls up to total ARR
    quarter_arr_sum = sum(q["arr"] for q in agg["by_quarter"])
    assert abs(quarter_arr_sum - agg["total_arr"]) < 0.01

    # Rivermark and Pinnacle are the engineered at-risk accounts
    by_id = {s["account_id"]: s for s in final["scored"]}
    assert by_id["acct-001"]["renew_probability"] < 0.85  # Pinnacle silent churn
    assert by_id["acct-011"]["renew_probability"] < 0.65  # Rivermark combined risk

    # Citadel and Aurora should be near-100%
    assert by_id["acct-012"]["renew_probability"] > 0.95
    assert by_id["acct-010"]["renew_probability"] > 0.95

    # PDF rendered
    pdf = tmp_path / "test-run-forecast.pdf"
    assert pdf.exists() and pdf.read_bytes()[:4] == b"%PDF"
