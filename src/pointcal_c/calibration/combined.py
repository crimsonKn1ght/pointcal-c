"""The one combined confidence score (hypothesis 3).

Pre-declared form: a two-feature logistic model of *correctness*,

    s = sigmoid(w0 + w1 * logit(p_cal) + w2 * d)

where ``p_cal`` is the temperature-scaled maximum softmax probability and ``d``
is normalized cross-view disagreement. Fit on clean calibration objects only, by
maximum likelihood against the binary correctness indicator.

Two properties matter and are enforced:

* the score is a *ranking* over predictions, never a class decision: the
  predicted class always comes from the fixed aggregated logits;
* nothing corrupted, and no evaluation object, is ever seen during the fit.

Fitting against correctness (rather than against the true-class likelihood) is
deliberate: selective prediction only needs a good ordering of "will this be
right", and the resulting score is directly interpretable as an estimated
probability of correctness.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


@dataclass
class CombinedScorer:
    """Fitted logistic blend of calibrated confidence and disagreement."""

    bias: float
    weight_confidence: float
    weight_disagreement: float
    num_samples: int = 0
    nll: float = float("nan")
    converged: bool = True

    @classmethod
    def fit(
        cls,
        confidence: np.ndarray,
        disagreement: np.ndarray,
        correct: np.ndarray,
        max_iter: int = 200,
        l2: float = 1e-4,
    ) -> "CombinedScorer":
        """Fit on clean calibration samples.

        Args:
            confidence: ``(n,)`` temperature-scaled max softmax probability.
            disagreement: ``(n,)`` normalized cross-view disagreement.
            correct: ``(n,)`` 1 where the fixed prediction was right.
            l2: small ridge term; keeps the fit finite when calibration
                accuracy is degenerate (all-correct or all-wrong).
        """
        confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
        disagreement = np.asarray(disagreement, dtype=np.float64).reshape(-1)
        target = np.asarray(correct, dtype=np.float64).reshape(-1)
        if not (confidence.shape == disagreement.shape == target.shape):
            raise ValueError("confidence, disagreement and correct must have the same length")
        if confidence.size == 0:
            raise ValueError("cannot fit the combined score on zero samples")

        features = torch.as_tensor(
            np.stack([_logit(confidence), disagreement], axis=1), dtype=torch.float64
        )
        y = torch.as_tensor(target, dtype=torch.float64)
        params = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [params], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe"
        )
        bce = torch.nn.functional.binary_cross_entropy_with_logits

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            z = params[0] + features @ params[1:]
            loss = bce(z, y) + l2 * (params[1:] ** 2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        w = params.detach().numpy()
        if not np.all(np.isfinite(w)):
            # Degenerate calibration set: fall back to ranking by confidence.
            w = np.array([0.0, 1.0, 0.0])
            converged = False
        else:
            converged = True

        with torch.no_grad():
            z = torch.as_tensor(w[0]) + features @ torch.as_tensor(w[1:])
            nll = float(bce(z, y))

        return cls(
            bias=float(w[0]),
            weight_confidence=float(w[1]),
            weight_disagreement=float(w[2]),
            num_samples=int(target.size),
            nll=nll,
            converged=converged,
        )

    def score(self, confidence: np.ndarray, disagreement: np.ndarray) -> np.ndarray:
        """Estimated probability of correctness, in ``[0, 1]``."""
        z = (
            self.bias
            + self.weight_confidence * _logit(confidence)
            + self.weight_disagreement * np.asarray(disagreement, dtype=np.float64)
        )
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path
