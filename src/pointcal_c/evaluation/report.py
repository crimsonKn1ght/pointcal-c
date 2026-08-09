"""Auto-generated results summary.

This writes the *observed evidence* half of the technical note: tables and
hypothesis checks generated mechanically from ``results.json``, so no number in
the write-up is transcribed by hand. Interpretation, limitations and related
work stay in ``docs/technical_note.md``, written by a human.

The hypothesis section reports what the numbers show, including when they show
the pre-registered hypothesis is wrong. A negative result is a result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pointcal_c import constants
from pointcal_c.config import RunConfig

_METHODS = constants.CONFIDENCE_METHODS
_HEADLINE_METRICS = ("accuracy", "ece", "nll", "brier", "aurc", "selective_risk@0.9")


def _rows(results_dir: Path) -> list[dict]:
    path = results_dir / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `pointcal-c evaluate` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _lookup(rows: list[dict], scope: str, value: str, method: str, metric: str) -> dict | None:
    for row in rows:
        if (
            row["scope"] == scope
            and str(row["scope_value"]) == str(value)
            and row["method"] == method
            and row["metric"] == metric
        ):
            return row
    return None


def _fmt(row: dict | None, digits: int = 4) -> str:
    if row is None or row.get("value") is None or not np.isfinite(float(row["value"])):
        return "n/a"
    value = float(row["value"])
    lo, hi = row.get("ci_lo"), row.get("ci_hi")
    if lo is not None and hi is not None and np.isfinite(float(lo)) and np.isfinite(float(hi)):
        return f"{value:.{digits}f} [{float(lo):.{digits}f}, {float(hi):.{digits}f}]"
    return f"{value:.{digits}f}"


def _table(rows: list[dict], scope: str, value: str, title: str) -> list[str]:
    lines = [f"### {title}", "", "| method | " + " | ".join(_HEADLINE_METRICS) + " |",
             "|" + "---|" * (len(_HEADLINE_METRICS) + 1)]
    for method in _METHODS:
        cells = [_fmt(_lookup(rows, scope, value, method, m)) for m in _HEADLINE_METRICS]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _hypothesis_section(rows: list[dict]) -> list[str]:
    lines = ["## Pre-registered hypothesis checks", "",
             "Mechanical checks against the pre-registered predictions. "
             "These are observations, not interpretations.", ""]

    def val(scope, value, method, metric):
        row = _lookup(rows, scope, value, method, metric)
        return float(row["value"]) if row and row.get("value") is not None else float("nan")

    # H1: severity worsens accuracy, ECE, NLL, Brier, AURC.
    lines.append("**H1 - severity degrades accuracy and reliability.**", )
    lines.append("")
    lines.append("| severity | accuracy | ECE | NLL | Brier | AURC |")
    lines.append("|---|---|---|---|---|---|")
    clean_cells = [val("overall", "clean", "temperature", m) for m in ("accuracy", "ece", "nll", "brier", "aurc")]
    lines.append("| clean | " + " | ".join(f"{c:.4f}" for c in clean_cells) + " |")
    for severity in constants.SEVERITIES:
        cells = [val("severity", str(severity), "temperature", m) for m in ("accuracy", "ece", "nll", "brier", "aurc")]
        lines.append(f"| {severity} | " + " | ".join(f"{c:.4f}" for c in cells) + " |")
    accs = [val("severity", str(s), "temperature", "accuracy") for s in constants.SEVERITIES]
    monotone = all(a >= b - 1e-9 for a, b in zip(accs, accs[1:]))
    lines += ["", f"Accuracy is monotonically non-increasing in severity: **{monotone}**.", ""]

    # H2: clean-only temperature improves calibration but not fully under shift.
    clean_msp = val("overall", "clean", "msp", "ece")
    clean_temp = val("overall", "clean", "temperature", "ece")
    corr_msp = val("overall", "corrupted", "msp", "ece")
    corr_temp = val("overall", "corrupted", "temperature", "ece")
    lines += [
        "**H2 - clean-fit temperature helps on average but does not fix shift.**",
        "",
        f"- clean ECE: {clean_msp:.4f} (MSP) -> {clean_temp:.4f} (temperature)",
        f"- corrupted ECE: {corr_msp:.4f} (MSP) -> {corr_temp:.4f} (temperature)",
        f"- improves on corrupted data: **{corr_temp < corr_msp}**; "
        f"residual corrupted ECE above clean: **{corr_temp - clean_temp:+.4f}**",
        "",
    ]

    # H3: combined score beats MSP on AURC and selective risk.
    lines += ["**H3 - combined score reduces selective risk versus raw MSP.**", "",
              "| metric | MSP | temperature | disagreement | combined | combined - MSP |",
              "|---|---|---|---|---|---|"]
    for metric in ("aurc", "excess_aurc", *(f"selective_risk@{c:g}" for c in constants.COVERAGE_LEVELS)):
        values = {m: val("overall", "corrupted", m, metric) for m in _METHODS}
        delta = values["combined"] - values["msp"]
        lines.append(
            f"| {metric} | " + " | ".join(f"{values[m]:.4f}" for m in _METHODS) + f" | {delta:+.4f} |"
        )
    aurc_delta = val("overall", "corrupted", "combined", "aurc") - val("overall", "corrupted", "msp", "aurc")
    lines += [
        "",
        f"Combined score lowers AURC relative to MSP on corrupted data: **{aurc_delta < 0}** "
        f"(negative delta means improvement).",
        "",
        "> Whether these gaps are meaningful must be read against the bootstrap "
        "intervals in `results.csv`, not from the point estimates alone.",
        "",
    ]
    return lines


def write_summary(cfg: RunConfig) -> Path:
    """Write ``results_summary.md`` into the run's results directory."""
    results_dir = Path(cfg.results_dir)
    rows = _rows(results_dir)

    lines = [
        "# PointCal-C results summary (auto-generated)",
        "",
        f"- tier: `{cfg.tier}`",
        f"- spec hash: `{constants.spec_hash()}`",
        f"- backbone: frozen OpenCLIP {cfg.model.arch} / {cfg.model.pretrained}",
        f"- aggregation: {constants.VIEW_AGGREGATION} over {constants.NUM_VIEWS} views",
        f"- disagreement statistic: {constants.DISAGREEMENT_PRIMARY}",
        "",
        "All intervals are 95% percentile bootstrap, resampled over base object IDs.",
        "NLL and Brier are undefined for the disagreement and combined scores, which",
        "rank predictions rather than define a distribution over the 40 classes.",
        "",
    ]

    ledger_path = Path(cfg.run_dir) / "ledger_inference.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        lines += [
            "## Measured compute",
            "",
            "| GPU | $/hr | GPU-hours | USD | views/s | peak VRAM (GB) | peak RAM (GB) |",
            "|---|---|---|---|---|---|---|",
            "| {gpu} | {hourly_usd:.2f} | {gpu_hours:.3f} | {usd:.3f} | "
            "{throughput_items_per_s:.1f} | {peak_vram_gb:.1f} | {peak_ram_gb:.1f} |".format(**ledger),
            "",
        ]

    calibration_path = results_dir / "calibration.json"
    if calibration_path.exists():
        bundle = json.loads(calibration_path.read_text(encoding="utf-8"))
        lines += [
            "## Fitted post-hoc parameters (clean calibration split only)",
            "",
            f"- temperature: **{bundle['temperature']['temperature']:.4f}** "
            f"(NLL {bundle['temperature']['nll_before']:.4f} -> {bundle['temperature']['nll_after']:.4f} "
            f"on {bundle['temperature']['num_samples']} samples)",
            f"- combined score: bias {bundle['combined']['bias']:+.4f}, "
            f"w_confidence {bundle['combined']['weight_confidence']:+.4f}, "
            f"w_disagreement {bundle['combined']['weight_disagreement']:+.4f}",
            f"- calibration-split accuracy: {bundle['calibration_accuracy']:.4f}",
            f"- fit time: {bundle['fit_seconds']:.2f}s (gate: 600s)",
            "",
            "Three scalars in total. The backbone contributes zero fitted parameters.",
            "",
        ]
        if bundle.get("degenerate"):
            lines += [
                "> **Warning: the calibration fit is degenerate.** The calibration set is "
                "close to all-correct or all-wrong, or the temperature hit a sanity "
                "bound, so the fitted scalars carry little information. Every "
                "calibration and combined-score number below should be read as "
                "unreliable, and the calibration split size revisited, before this "
                "run is reported.",
                "",
            ]

    lines += ["## Headline results", ""]
    lines += _table(rows, "overall", "clean", "Clean (evaluation objects)")
    lines += _table(rows, "overall", "corrupted", "All corrupted conditions pooled")

    lines += ["### By corruption family (all severities)", "",
              "| family | " + " | ".join(f"{m} acc" for m in _METHODS[:1]) +
              " | ECE (temp) | AURC (MSP) | AURC (combined) |", "|---|---|---|---|---|"]
    for family in constants.CORRUPTION_FAMILIES:
        lines.append(
            f"| {family} | "
            f"{_fmt(_lookup(rows, 'family', family, 'msp', 'accuracy'))} | "
            f"{_fmt(_lookup(rows, 'family', family, 'temperature', 'ece'))} | "
            f"{_fmt(_lookup(rows, 'family', family, 'msp', 'aurc'))} | "
            f"{_fmt(_lookup(rows, 'family', family, 'combined', 'aurc'))} |"
        )
    lines.append("")

    lines += _hypothesis_section(rows)
    lines += [
        "## Files",
        "",
        "- `results.csv` / `results.json` - full metrics table with intervals",
        "- `ablations.csv` - view count, prompt mode, and disagreement-statistic ablations",
        "- `predictions.npz` - per-sample confidences and correctness",
        "- `calibration.json` - the three fitted scalars",
        "- `examples.json` - confidently wrong / correctly abstained / high-disagreement cases",
        "",
    ]

    path = results_dir / "results_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
