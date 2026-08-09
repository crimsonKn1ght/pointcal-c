"""Compute-contract tests.

The contract is only worth anything if it fails closed, so these tests check
the refusals: over-price hardware, blown GPU-hour and dollar gates, memory
ceilings, and a projected runtime that would exceed four hours.
"""

from __future__ import annotations

import json

import pytest

from pointcal_c.budget import (
    GPU_HOURLY_USD,
    HOURLY_CEILING_USD,
    BudgetExceeded,
    BudgetGuard,
    assert_calibration_runtime,
    hourly_rate,
)
from pointcal_c.config import load_config


def test_allowed_gpus_are_all_under_the_ceiling():
    for gpu, rate in GPU_HOURLY_USD.items():
        assert rate <= HOURLY_CEILING_USD, gpu
        assert hourly_rate(gpu) == rate


def test_primary_gpu_is_the_a5000_at_the_listed_price():
    assert GPU_HOURLY_USD["RTX A5000"] == 0.27


@pytest.mark.parametrize("gpu", ["RTX 4090", "H100", "A100 80GB", ""])
def test_expensive_or_unknown_hardware_is_refused(gpu):
    with pytest.raises(BudgetExceeded):
        hourly_rate(gpu)


def test_tier_budget_cannot_exceed_the_project_cap():
    with pytest.raises(BudgetExceeded, match="6 GPU-hour"):
        BudgetGuard(gpu="RTX A5000", tier="full", max_gpu_hours=7.0, max_usd=1.0)


def test_gpu_hour_gate_trips():
    guard = BudgetGuard(
        gpu="RTX A5000", tier="xs", max_gpu_hours=1e-9, max_usd=10.0, billable=True
    )
    with pytest.raises(BudgetExceeded, match="GPU-h"), guard:
        guard.tick(items=1)


def test_dollar_gate_trips():
    guard = BudgetGuard(
        gpu="RTX A5000", tier="xs", max_gpu_hours=1.0, max_usd=1e-12, billable=True
    )
    with pytest.raises(BudgetExceeded, match=r"\$"), guard:
        guard.tick(items=1)


def test_ram_ceiling_trips():
    guard = BudgetGuard(
        gpu="RTX A5000", tier="xs", max_gpu_hours=1.0, max_usd=1.0,
        max_ram_gb=0.0, billable=False,
    )
    with pytest.raises(BudgetExceeded, match="peak RAM"), guard:
        guard.tick(items=1)


def test_projected_runtime_gate_trips_before_the_money_is_spent():
    guard = BudgetGuard(
        gpu="RTX A5000", tier="full", max_gpu_hours=6.0, max_usd=2.0,
        max_wall_hours=4.0, total_items=1000, billable=False,
    )
    with guard:
        guard._start -= 3600.0  # pretend an hour has already elapsed
        with pytest.raises(BudgetExceeded, match="projected full runtime"):
            guard.tick(items=100)  # 10% done after 1 h -> 10 h projected


def test_non_billable_runs_skip_the_spend_gates():
    guard = BudgetGuard(
        gpu="RTX A5000", tier="local", max_gpu_hours=1e-9, max_usd=1e-12, billable=False
    )
    with guard:
        guard.tick(items=10)
    assert guard.summary().usd == 0.0


def test_ledger_records_what_the_acceptance_criteria_ask_for(tmp_path):
    guard = BudgetGuard(gpu="A40", tier="s", max_gpu_hours=2.0, max_usd=1.0)
    with guard:
        guard.tick(items=64, label="clean")
    ledger = guard.summary()
    path = tmp_path / "ledger.json"
    ledger.write(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in ("gpu", "hourly_usd", "gpu_hours", "usd", "peak_vram_gb", "peak_ram_gb"):
        assert field in payload
    assert payload["hourly_usd"] == GPU_HOURLY_USD["A40"]
    assert payload["checkpoints"][0]["label"] == "clean"


def test_calibration_runtime_gate():
    assert_calibration_runtime(59.0)
    with pytest.raises(BudgetExceeded, match="10 min"):
        assert_calibration_runtime(601.0)


# --------------------------------------------------------------------------
# Shipped configs must satisfy the contract as written.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,gpu_hours,usd", [("xs", 0.5, 0.15), ("s", 2.0, 0.54)])
def test_shipped_tier_configs_match_the_ticket_gates(name, gpu_hours, usd):
    cfg = load_config(f"configs/{name}.yaml")
    assert cfg.budget.max_gpu_hours == gpu_hours
    assert cfg.budget.max_usd == usd
    assert cfg.budget.gpu == "RTX A5000"
    assert cfg.budget.max_wall_hours <= 4.0


def test_all_tiers_together_stay_inside_the_project_cap():
    totals = [load_config(f"configs/{t}.yaml") for t in ("xs", "s", "full")]
    assert sum(c.budget.max_gpu_hours for c in totals) <= 6.0
    assert sum(c.budget.max_usd for c in totals) <= 1.62


def test_full_config_covers_the_whole_benchmark():
    from pointcal_c import constants

    cfg = load_config("configs/full.yaml")
    assert set(cfg.conditions.corruptions) == set(constants.CORRUPTIONS)
    assert set(cfg.conditions.severities) == set(constants.SEVERITIES)
    assert len(cfg.conditions.pairs()) == 76  # 15 x 5 corrupted + clean
