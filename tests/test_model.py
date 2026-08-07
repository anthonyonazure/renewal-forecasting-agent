"""Tests for the deterministic logistic-regression scorer."""

from renewal.model import (
    _sigmoid,
    features_from_account,
    load_coefficients,
    score_account,
)


def test_sigmoid_at_zero_is_half():
    assert abs(_sigmoid(0) - 0.5) < 1e-9


def test_sigmoid_monotonic():
    assert _sigmoid(2) > _sigmoid(1) > _sigmoid(0) > _sigmoid(-1) > _sigmoid(-2)


def test_features_extraction():
    a = {
        "id": "x",
        "name": "x",
        "engagement_decay": 0.5,
        "avg_csat_3mo": 4.5,
        "p1_count_30d": 2,
        "usage_growth_pct": 50,
        "module_gap_count": 1,
        "exec_sponsor_present": 1,
        "previous_renewal_count": 0,
        "contract_months_in": 6,
    }
    f = features_from_account(a)
    assert f["usage_growth_pct_div_100"] == 0.5  # rescaled
    assert f["engagement_decay"] == 0.5


def test_score_healthy_account_high_prob():
    coefs = load_coefficients()
    healthy = {
        "id": "h",
        "name": "h",
        "tier": "gold",
        "arr": 100000,
        "contract_end": "2027-04-01",
        "engagement_decay": 0.0,
        "avg_csat_3mo": 4.8,
        "p1_count_30d": 0,
        "usage_growth_pct": 80,
        "module_gap_count": 0,
        "exec_sponsor_present": 1,
        "previous_renewal_count": 1,
        "contract_months_in": 8,
    }
    s = score_account(healthy, coefs)
    assert s["renew_probability"] > 0.95


def test_score_silent_churner_low_prob():
    coefs = load_coefficients()
    churner = {
        "id": "c",
        "name": "c",
        "tier": "gold",
        "arr": 100000,
        "contract_end": "2026-09-01",
        "engagement_decay": 0.85,
        "avg_csat_3mo": 4.0,
        "p1_count_30d": 0,
        "usage_growth_pct": -60,
        "module_gap_count": 0,
        "exec_sponsor_present": 0,
        "previous_renewal_count": 0,
        "contract_months_in": 14,
    }
    s = score_account(churner, coefs)
    assert s["renew_probability"] < 0.85


def test_arr_at_risk_calculation():
    coefs = load_coefficients()
    a = {
        "id": "a",
        "name": "a",
        "tier": "gold",
        "arr": 200000,
        "contract_end": "2027-01-01",
        "engagement_decay": 0.5,
        "avg_csat_3mo": 4.0,
        "p1_count_30d": 1,
        "usage_growth_pct": 0,
        "module_gap_count": 0,
        "exec_sponsor_present": 1,
        "previous_renewal_count": 0,
        "contract_months_in": 10,
    }
    s = score_account(a, coefs)
    expected = round((1 - s["renew_probability"]) * 200000, 2)
    assert s["arr_at_risk"] == expected


def test_contributions_sum_close_to_z():
    coefs = load_coefficients()
    a = {
        "id": "a",
        "name": "a",
        "tier": "gold",
        "arr": 100000,
        "contract_end": "2027-01-01",
        "engagement_decay": 0.3,
        "avg_csat_3mo": 4.5,
        "p1_count_30d": 0,
        "usage_growth_pct": 20,
        "module_gap_count": 0,
        "exec_sponsor_present": 1,
        "previous_renewal_count": 0,
        "contract_months_in": 6,
    }
    s = score_account(a, coefs)
    z_recomputed = coefs["intercept"] + sum(s["contributions"].values())
    from renewal.model import _sigmoid as sig

    assert abs(sig(z_recomputed) - s["renew_probability"]) < 1e-3
