"""Prompt construction from the frozen ensemble.

The ensemble in ``constants.PROMPT_ENSEMBLE`` is pre-declared and never searched
on evaluation data. Two prompt modes exist, and both are fixed before any
evaluation logits are computed:

* ``ensemble`` -- all templates; per-class text embeddings are averaged after
  L2 normalization and renormalized (the standard CLIP zero-shot classifier);
* ``canonical`` -- ``PROMPT_ENSEMBLE[0]`` alone, used only for the required
  prompt-ensemble-versus-single-prompt ablation.

The ablation needs a second text classifier but no second image pass: image
embeddings are cached per view, so both prompt modes are scored offline from
the same forward pass.
"""

from __future__ import annotations

import json

from pointcal_c import constants
from pointcal_c.provenance import sha256_bytes

PROMPT_MODES = ("ensemble", "canonical")


def build_prompts(mode: str = "ensemble") -> tuple[tuple[str, ...], ...]:
    """Return ``(num_classes, num_templates)`` prompt strings.

    Args:
        mode: ``"ensemble"`` or ``"canonical"``.
    """
    if mode not in PROMPT_MODES:
        raise ValueError(f"prompt mode must be one of {PROMPT_MODES}, got {mode!r}")
    templates = constants.PROMPT_ENSEMBLE if mode == "ensemble" else (constants.CANONICAL_PROMPT,)
    return tuple(
        tuple(template.format(name) for template in templates)
        for name in constants.class_prompt_names()
    )


def prompt_fingerprint(mode: str = "ensemble") -> str:
    """Hash of the exact prompt strings used, for the run manifest."""
    return sha256_bytes(json.dumps(build_prompts(mode), separators=(",", ":")).encode())
