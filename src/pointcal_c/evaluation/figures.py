"""The four core figures.

1. accuracy vs severity, per corruption family;
2. ECE vs severity, per confidence method;
3. risk-coverage curves on the pooled corrupted set, all four methods;
4. cost and throughput against the contract ceilings.

All four are rebuilt from written result files, so they can be regenerated
without touching a GPU or rerunning evaluation. matplotlib is imported lazily:
the analysis pipeline does not depend on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pointcal_c import constants
from pointcal_c.budget import GPU_HOURLY_USD, PROJECT_GPU_HOUR_CAP, PROJECT_USD_EXPECTED_A5000
from pointcal_c.evaluation.metrics import risk_coverage_curve

_METHOD_LABELS = {
    "msp": "max softmax",
    "temperature": "temperature (clean-fit)",
    "disagreement": "cross-view disagreement",
    "combined": "combined (clean-fit)",
}


def _load_rows(results_dir: Path) -> list[dict]:
    path = results_dir / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `pointcal-c evaluate` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _pick(rows: list[dict], **conditions) -> list[dict]:
    return [
        r
        for r in rows
        if all(str(r.get(k, "")) == str(v) for k, v in conditions.items())
    ]


def _value(rows: list[dict], default: float = float("nan")) -> float:
    if not rows:
        return default
    value = rows[0].get("value")
    return float(value) if value is not None else default


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    return paths


def figure_accuracy_vs_severity(rows: list[dict], out_dir: Path, method: str = "temperature"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    clean = _value(_pick(rows, scope="overall", scope_value="clean", method=method, metric="accuracy"))
    severities = [0, *constants.SEVERITIES]
    for family in constants.CORRUPTION_FAMILIES:
        ys = [clean]
        for severity in constants.SEVERITIES:
            ys.append(
                _value(
                    _pick(
                        rows,
                        scope="family_severity",
                        scope_value=f"{family}_s{severity}",
                        method=method,
                        metric="accuracy",
                    )
                )
            )
        ax.plot(severities, ys, marker="o", label=family)
    ax.axhline(clean, linestyle=":", linewidth=1, color="0.4")
    ax.annotate("clean", (0, clean), textcoords="offset points", xytext=(4, 6), fontsize=8, color="0.3")
    ax.set_xlabel("corruption severity (0 = clean)")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Zero-shot accuracy under corruption")
    ax.set_xticks(severities)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    return _save(fig, out_dir, "fig1_accuracy_vs_severity")


def figure_ece_vs_severity(rows: list[dict], out_dir: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    severities = [0, *constants.SEVERITIES]
    for method in constants.CONFIDENCE_METHODS:
        ys = [_value(_pick(rows, scope="overall", scope_value="clean", method=method, metric="ece"))]
        for severity in constants.SEVERITIES:
            ys.append(
                _value(_pick(rows, scope="severity", scope_value=str(severity), method=method, metric="ece"))
            )
        ax.plot(severities, ys, marker="s", label=_METHOD_LABELS.get(method, method))
    ax.set_xlabel("corruption severity (0 = clean)")
    ax.set_ylabel("expected calibration error")
    ax.set_title("Calibration degrades with corruption")
    ax.set_xticks(severities)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    return _save(fig, out_dir, "fig2_ece_vs_severity")


def figure_risk_coverage(results_dir: Path, out_dir: Path):
    import matplotlib.pyplot as plt

    path = results_dir / "predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `pointcal-c evaluate` first")
    with np.load(path, allow_pickle=False) as data:
        corrupted = data["condition"] != constants.CLEAN_CONDITION
        correct = data["correct"][corrupted]
        confidences = {
            key[len("confidence_") :]: data[key][corrupted]
            for key in data.files
            if key.startswith("confidence_")
        }

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for method in constants.CONFIDENCE_METHODS:
        if method not in confidences:
            continue
        coverage, risk = risk_coverage_curve(confidences[method], correct)
        ax.plot(coverage, risk, label=_METHOD_LABELS.get(method, method), linewidth=1.4)
    for level in constants.COVERAGE_LEVELS:
        ax.axvline(level, linestyle=":", linewidth=0.8, color="0.6")
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk (error rate on accepted)")
    ax.set_title("Risk-coverage, all corrupted conditions pooled")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    return _save(fig, out_dir, "fig3_risk_coverage")


def figure_cost_throughput(run_dir: Path, out_dir: Path):
    import matplotlib.pyplot as plt

    ledger_path = Path(run_dir) / "ledger_inference.json"
    if not ledger_path.exists():
        raise FileNotFoundError(f"{ledger_path} not found; run `pointcal-c infer` first")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    gpu_hours = float(ledger.get("gpu_hours", 0.0))
    usd = float(ledger.get("usd", 0.0))
    axes[0].bar(["GPU-hours", "USD"], [gpu_hours, usd], color=["#4C72B0", "#55A868"])
    axes[0].axhline(PROJECT_GPU_HOUR_CAP, linestyle="--", color="#C44E52", linewidth=1)
    axes[0].axhline(PROJECT_USD_EXPECTED_A5000, linestyle=":", color="#C44E52", linewidth=1)
    axes[0].set_title(
        f"Measured spend ({ledger.get('gpu', '?')} @ ${ledger.get('hourly_usd', 0):.2f}/hr)",
        fontsize=10,
    )
    axes[0].annotate(
        f"caps: {PROJECT_GPU_HOUR_CAP:g} GPU-h / ${PROJECT_USD_EXPECTED_A5000:.2f} expected",
        (0.02, 0.92), xycoords="axes fraction", fontsize=7.5, color="#C44E52",
    )
    for i, value in enumerate([gpu_hours, usd]):
        axes[0].annotate(f"{value:.3g}", (i, value), ha="center", va="bottom", fontsize=8)

    throughput = float(ledger.get("throughput_items_per_s", 0.0))
    peak_vram = float(ledger.get("peak_vram_gb", 0.0))
    peak_ram = float(ledger.get("peak_ram_gb", 0.0))
    axes[1].bar(
        ["views/s", "peak VRAM (GB)", "peak RAM (GB)"],
        [throughput, peak_vram, peak_ram],
        color=["#4C72B0", "#8172B2", "#CCB974"],
    )
    for i, value in enumerate([throughput, peak_vram, peak_ram]):
        axes[1].annotate(f"{value:.3g}", (i, value), ha="center", va="bottom", fontsize=8)
    axes[1].set_title("Throughput and peak memory", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.6, axis="y")
    fig.suptitle(
        f"Compute contract: allowed GPUs {', '.join(f'{k} ${v:.2f}/hr' for k, v in GPU_HOURLY_USD.items())}",
        fontsize=8,
        y=1.04,
    )
    return _save(fig, out_dir, "fig4_cost_throughput")


def make_all_figures(run_dir: str | Path, results_dir: str | Path, figures_dir: str | Path) -> list[str]:
    """Build every core figure that has the inputs it needs."""
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    rows = _load_rows(results_dir)
    written: list[str] = []
    for name, fn in (
        ("fig1", lambda: figure_accuracy_vs_severity(rows, figures_dir)),
        ("fig2", lambda: figure_ece_vs_severity(rows, figures_dir)),
        ("fig3", lambda: figure_risk_coverage(results_dir, figures_dir)),
        ("fig4", lambda: figure_cost_throughput(Path(run_dir), figures_dir)),
    ):
        try:
            written.extend(str(p) for p in fn())
        except FileNotFoundError as exc:
            print(f"  skipping {name}: {exc}")
    return written
