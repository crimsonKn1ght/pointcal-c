"""PointCal-C: low-cost selective zero-shot 3D recognition under corruption.

A reliability audit of frozen OpenCLIP applied to multi-view depth projections
of ModelNet40-C point clouds, with clean-only post-hoc calibration and a
multi-view-disagreement abstention baseline. The backbone is never trained.

See docs/ for the preregistration, provenance record, and compute contract.
"""

__version__ = "0.1.0"

from pointcal_c.constants import spec_hash  # noqa: F401
