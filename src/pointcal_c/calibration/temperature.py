"""Clean-fit scalar temperature scaling (Guo et al., 2017).

One scalar, fit by minimizing negative log-likelihood on *clean calibration-split*
samples only. Corrupted data, corruption identity and severity never touch it --
that is the point of hypothesis 2: a clean-only temperature should improve
calibration on average yet fail to fully correct corruption shift.

Dividing logits by a positive scalar cannot change the argmax, so temperature
scaling provably leaves predictions untouched. :meth:`apply` asserts it anyway.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

# Sanity bounds. A calibration set that is (nearly) all-correct pushes the NLL
# optimum toward T -> 0, i.e. infinite sharpening, which is a degenerate fit
# rather than a calibrated one and would look catastrophic under shift. Hitting
# a bound is recorded, not hidden.
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0


@dataclass
class TemperatureScaler:
    """A fitted scalar temperature."""

    temperature: float
    nll_before: float = float("nan")
    nll_after: float = float("nan")
    num_samples: int = 0
    converged: bool = True
    clamped: bool = False

    @classmethod
    def fit(
        cls,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 200,
        init_log_temperature: float = 0.0,
    ) -> "TemperatureScaler":
        """Fit T > 0 minimizing NLL.

        Args:
            logits: ``(n, num_classes)`` aggregated logits from clean
                calibration objects.
            labels: ``(n,)`` ground-truth class indices.

        The optimization is over ``log T`` in float64 on CPU with L-BFGS, which
        keeps the fit deterministic and bounded to well under a second.
        """
        if logits.ndim != 2:
            raise ValueError(f"expected (n, num_classes) logits, got {logits.shape}")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("logits and labels disagree in length")
        if logits.shape[0] == 0:
            raise ValueError("cannot fit a temperature on zero samples")

        x = torch.as_tensor(logits, dtype=torch.float64)
        y = torch.as_tensor(labels, dtype=torch.long)
        log_t = torch.tensor(float(init_log_temperature), dtype=torch.float64, requires_grad=True)
        loss_fn = torch.nn.functional.cross_entropy

        nll_before = float(loss_fn(x, y))
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = loss_fn(x / log_t.exp(), y)
            loss.backward()
            return loss

        optimizer.step(closure)

        temperature = float(log_t.detach().exp())
        if not np.isfinite(temperature) or temperature <= 0:
            # Degenerate fit: fall back to the identity rather than emit nonsense.
            return cls(
                temperature=1.0, nll_before=nll_before, nll_after=nll_before,
                num_samples=int(logits.shape[0]), converged=False,
            )

        clamped = not (MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE)
        if clamped:
            temperature = float(np.clip(temperature, MIN_TEMPERATURE, MAX_TEMPERATURE))
        nll_after = float(loss_fn(x / temperature, y))
        return cls(
            temperature=temperature,
            nll_before=nll_before,
            nll_after=nll_after,
            num_samples=int(logits.shape[0]),
            converged=bool(not clamped and nll_after <= nll_before + 1e-6),
            clamped=clamped,
        )

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Scale logits, asserting that predictions are unchanged."""
        scaled = np.asarray(logits, dtype=np.float64) / self.temperature
        if logits.size and not np.array_equal(np.argmax(logits, axis=-1), np.argmax(scaled, axis=-1)):
            raise AssertionError("temperature scaling changed a prediction; this must never happen")
        return scaled

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path
