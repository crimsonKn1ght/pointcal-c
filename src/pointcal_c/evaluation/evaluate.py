"""Evaluation driver: cached logits in, machine-readable results out.

Everything here runs on CPU from the logit cache. No CLIP forward pass, no GPU
time, no dollars -- which is why every ablation is affordable and why a mistake
in analysis costs nothing to fix.

Reporting scopes, per the ticket: overall (clean and corrupted), per corruption
family, per corruption type, per severity, and per individual condition. Every
scope is reported for all four confidence methods with object-grouped bootstrap
intervals, and every corrupted scope also carries its degradation relative to
clean.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pointcal_c import constants
from pointcal_c.calibration.fit import CalibrationBundle, fit_calibration
from pointcal_c.config import RunConfig
from pointcal_c.data.splits import Split, assert_evaluation_only
from pointcal_c.evaluation.bootstrap import grouped_bootstrap
from pointcal_c.evaluation.metrics import BOOTSTRAPPED_METRICS, metric_bundle
from pointcal_c.evaluation.scoring import ScoredCondition, score_condition
from pointcal_c.inference.cache import list_cached_conditions, load_cache

RESULT_FIELDS = (
    "config", "scope", "scope_value", "severity", "family",
    "method", "metric", "value", "ci_lo", "ci_hi", "n",
)


@dataclass
class Pooled:
    """All scored conditions concatenated into flat, mask-friendly arrays."""

    object_ids: np.ndarray
    labels: np.ndarray
    correct: np.ndarray
    condition: np.ndarray
    corruption: np.ndarray
    family: np.ndarray
    severity: np.ndarray  # -1 for clean
    confidence: dict[str, np.ndarray] = field(default_factory=dict)
    probs: dict[str, np.ndarray | None] = field(default_factory=dict)

    @classmethod
    def from_conditions(cls, scored: list[ScoredCondition]) -> "Pooled":
        methods = list(scored[0].scores)
        confidence = {
            m: np.concatenate([s.scores[m].confidence for s in scored]) for m in methods
        }
        probs: dict[str, np.ndarray | None] = {}
        for m in methods:
            if scored[0].scores[m].probs is None:
                probs[m] = None
            else:
                probs[m] = np.concatenate(
                    [s.scores[m].probs.astype(np.float32) for s in scored]
                )
        return cls(
            object_ids=np.concatenate([s.object_ids for s in scored]),
            labels=np.concatenate([s.labels for s in scored]),
            correct=np.concatenate([s.correct for s in scored]),
            condition=np.concatenate([np.full(s.num_samples, s.condition) for s in scored]),
            corruption=np.concatenate([np.full(s.num_samples, s.corruption) for s in scored]),
            family=np.concatenate([np.full(s.num_samples, s.family) for s in scored]),
            severity=np.concatenate(
                [np.full(s.num_samples, -1 if s.severity is None else s.severity) for s in scored]
            ),
            confidence=confidence,
            probs=probs,
        )

    def scope_masks(self) -> list[tuple[str, str, int | None, str | None, np.ndarray]]:
        """Every reporting scope as ``(scope, value, severity, family, mask)``."""
        is_clean = self.condition == constants.CLEAN_CONDITION
        scopes: list[tuple[str, str, int | None, str | None, np.ndarray]] = [
            ("overall", "clean", None, None, is_clean),
            ("overall", "corrupted", None, None, ~is_clean),
        ]
        for family in sorted(set(self.family[~is_clean].tolist())):
            scopes.append(("family", family, None, family, self.family == family))
        for corruption in sorted(set(self.corruption[~is_clean].tolist())):
            scopes.append(
                (
                    "corruption",
                    corruption,
                    None,
                    constants.CORRUPTION_TO_FAMILY[corruption],
                    self.corruption == corruption,
                )
            )
        for severity in sorted({int(s) for s in self.severity if s > 0}):
            scopes.append(("severity", str(severity), severity, None, self.severity == severity))
        # Family x severity: the grid the accuracy/ECE-vs-severity figures plot.
        for family in sorted(set(self.family[~is_clean].tolist())):
            for severity in sorted({int(s) for s in self.severity if s > 0}):
                mask = (self.family == family) & (self.severity == severity)
                if mask.any():
                    scopes.append(
                        ("family_severity", f"{family}_s{severity}", severity, family, mask)
                    )
        for condition in sorted(set(self.condition.tolist())):
            mask = self.condition == condition
            sev = int(self.severity[mask][0])
            scopes.append(
                (
                    "condition",
                    condition,
                    None if sev < 0 else sev,
                    str(self.family[mask][0]),
                    mask,
                )
            )
        return scopes


def _statistic_factory(pooled: Pooled, method: str, rows: np.ndarray):
    """Metric closure over a row subset, for both point estimate and bootstrap."""
    probs = pooled.probs[method]

    def statistic(sub: np.ndarray) -> dict[str, float | None]:
        # `sub` indexes positions within the scope, `rows` maps them to pooled rows.
        idx = rows[sub]
        return metric_bundle(
            correct=pooled.correct[idx],
            confidence=pooled.confidence[method][idx],
            labels=pooled.labels[idx],
            probs=None if probs is None else probs[idx],
        )

    return statistic


def evaluate_pooled(
    pooled: Pooled,
    config_key: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict]:
    """Point estimates plus grouped bootstrap intervals for every scope/method."""
    rows_out: list[dict] = []
    clean_values: dict[tuple[str, str], float] = {}

    for scope, value, severity, family, mask in pooled.scope_masks():
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            continue
        for method in constants.CONFIDENCE_METHODS:
            statistic = _statistic_factory(pooled, method, rows)
            point = statistic(np.arange(rows.size))
            intervals = (
                grouped_bootstrap(
                    pooled.object_ids[rows],
                    statistic,
                    num_samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                if bootstrap_samples > 0
                else {}
            )
            for metric, val in point.items():
                if metric == "n":
                    continue
                lo, hi = intervals.get(metric, (float("nan"), float("nan")))
                if metric not in BOOTSTRAPPED_METRICS:
                    lo = hi = float("nan")
                rows_out.append(
                    {
                        "config": config_key,
                        "scope": scope,
                        "scope_value": value,
                        "severity": "" if severity is None else severity,
                        "family": family or "",
                        "method": method,
                        "metric": metric,
                        "value": val,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "n": int(rows.size),
                    }
                )
                if scope == "overall" and value == "clean" and val is not None:
                    clean_values[(method, metric)] = float(val)

    # Degradation relative to clean, for every corrupted scope.
    deltas: list[dict] = []
    for row in rows_out:
        if row["scope"] == "overall" and row["scope_value"] == "clean":
            continue
        baseline = clean_values.get((row["method"], row["metric"]))
        if baseline is None or row["value"] is None or not np.isfinite(row["value"]):
            continue
        deltas.append(
            {
                **row,
                "metric": f"delta_{row['metric']}_vs_clean",
                "value": float(row["value"]) - baseline,
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
            }
        )
    return rows_out + deltas


def collect_examples(scored: list[ScoredCondition], per_bucket: int = 15) -> dict:
    """Qualitative cases required by the ticket.

    * confidently wrong: highest combined confidence among errors;
    * correctly abstained: lowest combined confidence among errors (an
      abstention policy would have declined exactly these);
    * view disagreement: highest cross-view divergence.
    """
    rows = []
    for cond in scored:
        for i in range(cond.num_samples):
            rows.append(
                {
                    "condition": cond.condition,
                    "object_id": int(cond.object_ids[i]),
                    "true_class": constants.MODELNET40_CLASSES[int(cond.labels[i])],
                    "predicted_class": constants.MODELNET40_CLASSES[int(cond.predictions[i])],
                    "correct": bool(cond.correct[i]),
                    "msp": float(cond.scores["msp"].confidence[i]),
                    "calibrated_confidence": float(cond.scores["temperature"].confidence[i]),
                    "combined_score": float(cond.scores["combined"].confidence[i]),
                    "disagreement": float(cond.disagreement[i]),
                    "view_agreement": float(cond.view_agreement[i]),
                    "blank_views": int(cond.blank_views[i]),
                }
            )
    errors = [r for r in rows if not r["correct"]]
    return {
        "confidently_wrong": sorted(errors, key=lambda r: -r["combined_score"])[:per_bucket],
        "correctly_abstained": sorted(errors, key=lambda r: r["combined_score"])[:per_bucket],
        "high_view_disagreement": sorted(rows, key=lambda r: -r["disagreement"])[:per_bucket],
        "blank_view_cases": [r for r in rows if r["blank_views"] > 0][:per_bucket],
    }


def write_results(rows: list[dict], results_dir: str | Path, stem: str = "results") -> dict[str, Path]:
    """Emit the machine-readable metrics table as CSV and JSON."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{stem}.csv"
    json_path = results_dir / f"{stem}.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})
    json_path.write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}


def run_evaluation(
    cfg: RunConfig,
    split: Split,
    views: tuple[str, ...] = constants.VIEW_NAMES,
    prompt_mode: str = "ensemble",
    disagreement_statistic: str = constants.DISAGREEMENT_PRIMARY,
    bootstrap_samples: int | None = None,
    write: bool = True,
) -> dict:
    """Fit calibration, score every cached condition, and write the tables."""
    conditions = list_cached_conditions(cfg.logits_dir)
    if constants.CLEAN_CONDITION not in conditions:
        raise FileNotFoundError(
            "the clean condition must be cached before evaluation: calibration is clean-only"
        )

    clean_cache = load_cache(cfg.logits_dir, constants.CLEAN_CONDITION)
    bundle = fit_calibration(
        clean_cache,
        split,
        views=views,
        prompt_mode=prompt_mode,
        disagreement_statistic=disagreement_statistic,
    )

    eval_ids = np.intersect1d(clean_cache.object_ids, split.evaluation_array)
    assert_evaluation_only(eval_ids, split)

    scored: list[ScoredCondition] = []
    for condition in conditions:
        cache = load_cache(cfg.logits_dir, condition)
        scored.append(score_condition(cache, bundle, object_ids=eval_ids))

    pooled = Pooled.from_conditions(scored)
    n_boot = cfg.evaluation.bootstrap_samples if bootstrap_samples is None else bootstrap_samples
    rows = evaluate_pooled(pooled, bundle.key, n_boot, cfg.evaluation.bootstrap_seed)

    out: dict = {
        "config_key": bundle.key,
        "conditions": conditions,
        "num_evaluation_objects": int(eval_ids.size),
        "num_rows": int(pooled.labels.size),
        "bootstrap_samples": n_boot,
        "calibration": bundle.to_dict(),
        "rows": rows,
    }
    if write:
        paths = write_results(rows, cfg.results_dir)
        bundle.save(cfg.results_dir / "calibration.json")
        examples = collect_examples(scored)
        (cfg.results_dir / "examples.json").write_text(
            json.dumps(examples, indent=2), encoding="utf-8"
        )
        # Raw per-sample predictions, required in the final artifact.
        np.savez_compressed(
            cfg.results_dir / "predictions.npz",
            object_ids=pooled.object_ids,
            condition=pooled.condition,
            labels=pooled.labels,
            correct=pooled.correct,
            **{f"confidence_{m}": pooled.confidence[m].astype(np.float32) for m in pooled.confidence},
        )
        out["paths"] = {k: str(v) for k, v in paths.items()}
    return out
