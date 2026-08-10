"""The required ablations, all computed offline from the cached logits.

Ticket list:

* **1 / 3 / 6 views**: re-aggregate a view subset from the same cache.
* **temperature on / off**: the ``msp`` and ``temperature`` methods already
  differ by exactly this, so it is a method contrast, not a rerun.
* **disagreement on / off**: the ``temperature`` and ``combined`` methods
  differ by exactly the disagreement feature; the secondary statistic
  (logit variance) is run as an extra configuration.
* **prompt ensemble vs single canonical prompt**: both text classifiers were
  scored during the one image forward pass, so this is also free.

CLIP is never rerun. Each configuration refits its own clean-only calibration,
because changing the views or the prompt changes the logits being calibrated.
"""

from __future__ import annotations

import json
from pathlib import Path

from pointcal_c import constants
from pointcal_c.config import RunConfig
from pointcal_c.data.splits import Split
from pointcal_c.evaluation.evaluate import run_evaluation, write_results


def ablation_configurations() -> list[dict]:
    """The pre-declared ablation grid."""
    configs = [
        {
            "name": f"views{k}_ensemble",
            "views": constants.VIEW_SUBSETS[k],
            "prompt_mode": "ensemble",
            "disagreement_statistic": constants.DISAGREEMENT_PRIMARY,
        }
        for k in sorted(constants.VIEW_SUBSETS)
    ]
    configs.append(
        {
            "name": "views6_canonical_prompt",
            "views": constants.VIEW_NAMES,
            "prompt_mode": "canonical",
            "disagreement_statistic": constants.DISAGREEMENT_PRIMARY,
        }
    )
    configs.append(
        {
            "name": "views6_ensemble_logit_variance",
            "views": constants.VIEW_NAMES,
            "prompt_mode": "ensemble",
            "disagreement_statistic": constants.DISAGREEMENT_SECONDARY,
        }
    )
    return configs


def run_ablations(
    cfg: RunConfig,
    split: Split,
    bootstrap_samples: int | None = None,
    write: bool = True,
) -> dict:
    """Run every ablation configuration and write one combined table."""
    n_boot = (
        min(200, cfg.evaluation.bootstrap_samples)
        if bootstrap_samples is None
        else bootstrap_samples
    )
    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}

    for spec in ablation_configurations():
        result = run_evaluation(
            cfg,
            split,
            views=spec["views"],
            prompt_mode=spec["prompt_mode"],
            disagreement_statistic=spec["disagreement_statistic"],
            bootstrap_samples=n_boot,
            write=False,
        )
        for row in result["rows"]:
            row["config"] = spec["name"]
        all_rows.extend(result["rows"])
        summaries[spec["name"]] = {
            "views": list(spec["views"]),
            "prompt_mode": spec["prompt_mode"],
            "disagreement_statistic": spec["disagreement_statistic"],
            "calibration": result["calibration"],
        }

    out = {"configurations": summaries, "rows": all_rows, "bootstrap_samples": n_boot}
    if write:
        paths = write_results(all_rows, cfg.results_dir, stem="ablations")
        out["paths"] = {k: str(v) for k, v in paths.items()}
        (Path(cfg.results_dir) / "ablation_configs.json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
    return out
