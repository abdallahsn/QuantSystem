from __future__ import annotations

from modules.config_v19 import load_release_gates
from modules.release_gates_v19 import evaluate_release_gates


def _passing_metrics() -> dict:
    return {
        "n_folds": 3,
        "total_trades": 45,
        "mean_directional_f1": 0.40,
        "mean_directional_precision": 0.42,
        "mean_directional_recall": 0.31,
        "mean_profit_factor": 1.12,
        "mean_expectancy_dollars": 0.50,
        "mean_ece": 0.08,
        "max_drawdown_pct": 0.03,
        "mean_stress_profit_factor": 1.04,
        "mean_stress_expectancy_dollars": 0.10,
        "release_blocker_count": 0,
    }


def test_release_gates_pass_complete_metrics():
    report = evaluate_release_gates(_passing_metrics(), load_release_gates())

    assert report["passed"] is True
    assert all(result["passed"] for result in report["results"])


def test_release_gates_fail_missing_required_metric():
    metrics = _passing_metrics()
    metrics.pop("mean_profit_factor")

    report = evaluate_release_gates(metrics, load_release_gates())

    assert report["passed"] is False
    assert any(
        result["metric"] == "mean_profit_factor"
        and result["reason"] == "required_metric_missing"
        for result in report["results"]
    )


def test_release_gates_fail_low_quality_metrics():
    metrics = _passing_metrics()
    metrics.update(
        {
            "total_trades": 10,
            "mean_directional_f1": 0.10,
            "mean_profit_factor": 0.80,
            "mean_ece": 0.25,
            "max_drawdown_pct": 0.20,
            "mean_stress_profit_factor": 0.70,
        }
    )

    report = evaluate_release_gates(metrics, load_release_gates())
    failed = {result["metric"] for result in report["results"] if not result["passed"]}

    assert report["passed"] is False
    assert {
        "total_trades",
        "mean_directional_f1",
        "mean_profit_factor",
        "mean_ece",
        "max_drawdown_pct",
        "mean_stress_profit_factor",
    } <= failed


def test_release_gates_fail_fold_level_blockers():
    metrics = _passing_metrics()
    metrics["release_blocker_count"] = 1

    report = evaluate_release_gates(metrics, load_release_gates())

    assert report["passed"] is False
    assert any(result["metric"] == "release_blocker_count" for result in report["results"])
