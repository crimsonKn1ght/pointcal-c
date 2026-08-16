"""Compute contract enforcement.

GOU-101 fixes hardware, runtime and spend gates. This module makes them
executable rather than aspirational: an out-of-contract GPU refuses to start a
run, and a run that projects past its wall-clock or spend gate raises
``BudgetExceeded`` at the next checkpoint instead of quietly finishing.

The prescribed response to a breach is to fall back to the last passing tier.
Renting a larger GPU is explicitly not an option.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil
import torch

# Runpod Community Cloud list prices recorded 2026-08-09 (docs/provenance.md).
GPU_HOURLY_USD: dict[str, float] = {
    "RTX A5000": 0.27,   # primary
    "A40": 0.44,         # fallback
    "RTX 3090": 0.50,    # fallback
    "RTX A6000": 0.53,   # fallback
    # Added 2026-08-16: A5000/3090 were both "Out of capacity" on Community
    # Cloud at launch time. Used for a live/demo run, not the paper-track run.
    # See docs/provenance.md "Compute prices".
    "RTX 4000 Ada": 0.28,
}

# Hard machine-price ceiling from the ticket. The RTX 4090 ($0.69/hr) is out.
HOURLY_CEILING_USD = 0.60

PROJECT_GPU_HOUR_CAP = 6.0
PROJECT_USD_CAP_FALLBACK = 3.60
PROJECT_USD_EXPECTED_A5000 = 1.62
CALIBRATION_MINUTES_CAP = 10.0
SETUP_PAID_MINUTES_CAP = 45.0


class BudgetExceeded(RuntimeError):
    """Raised when a gate in the compute contract is breached."""


def hourly_rate(gpu: str) -> float:
    """Price for an allowed GPU. Unknown or over-ceiling hardware is refused."""
    if gpu not in GPU_HOURLY_USD:
        raise BudgetExceeded(
            f"GPU {gpu!r} is not on the allowed list {sorted(GPU_HOURLY_USD)}; "
            "do not launch hardware outside the contract"
        )
    rate = GPU_HOURLY_USD[gpu]
    if rate > HOURLY_CEILING_USD:
        raise BudgetExceeded(f"{gpu} at ${rate:.2f}/hr exceeds the ${HOURLY_CEILING_USD:.2f}/hr ceiling")
    return rate


def match_gpu(reported_name: str) -> str:
    """Map a driver-reported device name onto a contract entry.

    ``torch.cuda.get_device_name`` returns strings like ``"NVIDIA RTX A5000"``;
    the price table is keyed by the Runpod listing name. Anything that does not
    match an allowed entry is refused rather than guessed at.
    """
    cleaned = reported_name.replace("NVIDIA", "").replace("GeForce", "").strip()
    for gpu in GPU_HOURLY_USD:
        if cleaned.lower() == gpu.lower() or gpu.lower() in cleaned.lower():
            return gpu
    raise BudgetExceeded(
        f"detected GPU {reported_name!r} is not on the allowed list "
        f"{sorted(GPU_HOURLY_USD)}; stop the pod rather than running out of contract"
    )


def detect_gpu() -> tuple[str, float]:
    """``(contract name, $/hr)`` for the currently visible GPU."""
    if not torch.cuda.is_available():
        raise BudgetExceeded("no CUDA device visible")
    if torch.cuda.device_count() > 1:
        raise BudgetExceeded(
            f"{torch.cuda.device_count()} GPUs visible; the contract allows exactly one"
        )
    gpu = match_gpu(torch.cuda.get_device_name(0))
    return gpu, hourly_rate(gpu)


@dataclass
class Ledger:
    """The cost/resource record required by the acceptance criteria."""

    gpu: str
    hourly_usd: float
    tier: str
    gpu_name_reported: str = ""
    wall_seconds: float = 0.0
    gpu_hours: float = 0.0
    usd: float = 0.0
    peak_vram_gb: float = 0.0
    peak_ram_gb: float = 0.0
    items_processed: int = 0
    # Items served from an existing logit cache rather than recomputed. Counted
    # for runtime projection but kept out of items_processed, so throughput and
    # cost describe work actually done on this run.
    items_skipped_cached: int = 0
    throughput_items_per_s: float = 0.0
    checkpoints: list[dict] = field(default_factory=list)

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class BudgetGuard:
    """Tracks spend and projects completion; raises before money is wasted.

    Usage::

        guard = BudgetGuard(gpu="RTX A5000", tier="xs", max_gpu_hours=0.5,
                            max_usd=0.15, max_wall_hours=4.0, total_items=n)
        with guard:
            for batch in ...:
                ...
                guard.tick(len(batch))
    """

    def __init__(
        self,
        gpu: str,
        tier: str,
        max_gpu_hours: float,
        max_usd: float,
        max_wall_hours: float = 4.0,
        max_vram_gb: float = 20.0,
        max_ram_gb: float = 25.0,
        total_items: int | None = None,
        billable: bool = True,
    ) -> None:
        self.rate = hourly_rate(gpu) if billable else 0.0
        if billable and max_gpu_hours > PROJECT_GPU_HOUR_CAP:
            raise BudgetExceeded("tier budget exceeds the 6 GPU-hour project cap")
        self.gpu = gpu
        self.tier = tier
        self.max_gpu_hours = max_gpu_hours
        self.max_usd = max_usd
        self.max_wall_hours = max_wall_hours
        self.max_vram_gb = max_vram_gb
        self.max_ram_gb = max_ram_gb
        self.total_items = total_items
        self.billable = billable
        self.ledger = Ledger(gpu=gpu, hourly_usd=self.rate, tier=tier)
        self._start: float | None = None
        self._proc = psutil.Process()

    # -- lifecycle -------------------------------------------------------
    def __enter__(self) -> "BudgetGuard":
        self._start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.ledger.gpu_name_reported = torch.cuda.get_device_name(0)
        return self

    def __exit__(self, *exc) -> None:
        self._refresh()
        return None

    # -- accounting ------------------------------------------------------
    @property
    def elapsed_hours(self) -> float:
        if self._start is None:
            return 0.0
        return (time.perf_counter() - self._start) / 3600.0

    def _refresh(self) -> None:
        hours = self.elapsed_hours
        self.ledger.wall_seconds = hours * 3600.0
        self.ledger.gpu_hours = hours
        self.ledger.usd = hours * self.rate
        ram_gb = self._proc.memory_info().rss / 1024**3
        self.ledger.peak_ram_gb = max(self.ledger.peak_ram_gb, ram_gb)
        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1024**3
            self.ledger.peak_vram_gb = max(self.ledger.peak_vram_gb, vram_gb)
        if hours > 0 and self.ledger.items_processed:
            self.ledger.throughput_items_per_s = self.ledger.items_processed / (hours * 3600.0)

    def tick(self, items: int = 1, label: str = "") -> None:
        """Account for completed work and enforce every gate."""
        self.ledger.items_processed += items
        self._refresh()
        hours = self.elapsed_hours

        if self.billable and hours > self.max_gpu_hours:
            raise BudgetExceeded(
                f"{self.tier}: {hours:.3f} GPU-h exceeds the {self.max_gpu_hours:.2f} GPU-h gate"
            )
        if self.billable and self.ledger.usd > self.max_usd:
            raise BudgetExceeded(
                f"{self.tier}: ${self.ledger.usd:.2f} exceeds the ${self.max_usd:.2f} gate"
            )
        if hours > self.max_wall_hours:
            raise BudgetExceeded(f"{self.tier}: wall time exceeds {self.max_wall_hours:.1f} h")
        if self.ledger.peak_vram_gb > self.max_vram_gb:
            raise BudgetExceeded(
                f"peak VRAM {self.ledger.peak_vram_gb:.1f} GB exceeds the "
                f"{self.max_vram_gb:.0f} GB target; reduce batch size, do not rent a larger GPU"
            )
        if self.ledger.peak_ram_gb > self.max_ram_gb:
            raise BudgetExceeded(
                f"peak RAM {self.ledger.peak_ram_gb:.1f} GB exceeds the {self.max_ram_gb:.0f} GB target"
            )

        projected = self.projected_hours()
        if projected is not None and projected > self.max_wall_hours:
            raise BudgetExceeded(
                f"projected full runtime {projected:.2f} h exceeds the "
                f"{self.max_wall_hours:.1f} h gate at {self.ledger.items_processed}/"
                f"{self.total_items} items; fall back to the last passing tier"
            )
        if label:
            self.ledger.checkpoints.append(
                {
                    "label": label,
                    "items": self.ledger.items_processed,
                    "hours": round(hours, 4),
                    "usd": round(self.ledger.usd, 4),
                    "peak_vram_gb": round(self.ledger.peak_vram_gb, 2),
                    "peak_ram_gb": round(self.ledger.peak_ram_gb, 2),
                }
            )

    def projected_hours(self) -> float | None:
        """Extrapolated total runtime from measured throughput.

        Cache-served items count towards progress here (they are genuinely done)
        even though they cost no GPU time, so a resumed run does not project a
        false overrun from the work it still has left.
        """
        done = self.ledger.items_processed + self.ledger.items_skipped_cached
        if not self.total_items or done < max(8, self.total_items // 100):
            return None
        if not self.ledger.items_processed:
            return None  # nothing recomputed: elapsed time says nothing about rate
        frac = done / self.total_items
        return self.elapsed_hours / frac

    def summary(self) -> Ledger:
        self._refresh()
        return self.ledger


def assert_calibration_runtime(seconds: float) -> None:
    """Calibration 'training' must finish within 10 minutes (ticket gate)."""
    if seconds > CALIBRATION_MINUTES_CAP * 60:
        raise BudgetExceeded(
            f"calibration took {seconds / 60:.1f} min, over the "
            f"{CALIBRATION_MINUTES_CAP:.0f} min gate"
        )
