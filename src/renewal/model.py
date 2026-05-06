"""Deterministic logistic-regression scorer.

Pure Python — no scikit-learn dependency. The coefficients live in
model/coefficients.json (versioned, diffable, auditable). Each run
records the version used so the forecast is reproducible from the
account data alone.

Probability = 1 / (1 + exp(-(intercept + Σ coef[k] * feature[k])))

For features with implicit scaling (`usage_growth_pct_div_100` divides
the raw percentage by 100), the convention is to bake the scaling into
the coefficient name so it's obvious what's happening."""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_coefficients(path: str | None = None) -> dict:
    p = Path(path or "model/coefficients.json")
    if not p.exists():
        # Fallback to package-bundled coefficients
        p = Path(__file__).resolve().parents[2] / "model" / "coefficients.json"
    return json.loads(p.read_text())


def features_from_account(account: dict) -> dict[str, float]:
    return {
        "engagement_decay": float(account.get("engagement_decay", 0)),
        "avg_csat_3mo": float(account.get("avg_csat_3mo", 4.5)),
        "p1_count_30d": float(account.get("p1_count_30d", 0)),
        "usage_growth_pct_div_100": float(account.get("usage_growth_pct", 0)) / 100.0,
        "module_gap_count": float(account.get("module_gap_count", 0)),
        "exec_sponsor_present": float(account.get("exec_sponsor_present", 0)),
        "previous_renewal_count": float(account.get("previous_renewal_count", 0)),
        "contract_months_in": float(account.get("contract_months_in", 0)),
    }


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score_account(account: dict, coefs: dict) -> dict:
    features = features_from_account(account)
    coef_map = coefs["coefficients"]
    z = coefs["intercept"] + sum(coef_map[k] * features[k] for k in coef_map)
    prob = round(_sigmoid(z), 4)
    arr = float(account.get("arr", 0))
    return {
        "account_id": account["id"],
        "account_name": account["name"],
        "arr": arr,
        "tier": account.get("tier"),
        "contract_end": account["contract_end"],
        "renew_probability": prob,
        "arr_at_risk": round((1 - prob) * arr, 2),
        "model_version": coefs["version"],
        "features": features,
        # Per-feature contribution to z (for explanation)
        "contributions": {k: round(coef_map[k] * features[k], 4) for k in coef_map},
    }
