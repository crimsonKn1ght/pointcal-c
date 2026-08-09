"""Frozen OpenCLIP ViT-B/32 wrapper.

Hard rules from GOU-101, enforced here rather than documented:

* the backbone is loaded in eval mode with ``requires_grad_(False)``;
* no adapter, LoRA, prompt tuning, test-time adaptation or external LLM call;
* FP16 autocast on CUDA, FP32 on CPU (the local smoke path);
* text features are computed once per prompt mode and reused;
* the image encoder is called exactly once per (object, view); every prompt
  mode and view-count ablation is scored offline from the cached embedding.

``open_clip`` is imported lazily so that split construction, calibration,
evaluation and the whole test suite run in an environment without it.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from pointcal_c import constants
from pointcal_c.model.prompts import build_prompts, prompt_fingerprint
from pointcal_c.projection.depth_views import CLIP_PIXEL_MEAN, CLIP_PIXEL_STD
from pointcal_c.provenance import sha256_file


@dataclass
class BackboneInfo:
    arch: str
    pretrained: str
    device: str
    precision: str
    embed_dim: int
    logit_scale: float
    num_params: int
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    prompt_fingerprints: dict[str, str]


class FrozenCLIP:
    """Zero-shot classifier built on a frozen OpenCLIP image/text encoder."""

    def __init__(
        self,
        arch: str = constants.CLIP_ARCH,
        pretrained: str = constants.CLIP_PRETRAINED,
        device: str = "cuda",
        precision: str = "fp16",
        cache_dir: str | Path | None = None,
        pinned_checkpoint_sha256: str | None = None,
    ) -> None:
        if arch != constants.CLIP_ARCH or pretrained != constants.CLIP_PRETRAINED:
            raise ValueError("the backbone is frozen to OpenCLIP ViT-B/32 laion2b_s34b_b79k")
        try:
            import open_clip  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "open_clip is required for inference: pip install open_clip_torch. "
                "Every other stage of PointCal-C runs without it."
            ) from exc

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.precision = precision if self.device.type == "cuda" else "fp32"
        self.arch = arch
        self.pretrained = pretrained

        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, cache_dir=str(cache_dir) if cache_dir else None
        )
        model.eval().requires_grad_(False)
        self.model = model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(arch)

        self._assert_normalization_matches(preprocess)
        self.checkpoint_path = self._resolve_checkpoint(open_clip, arch, pretrained, cache_dir)
        self.checkpoint_sha256 = (
            sha256_file(self.checkpoint_path) if self.checkpoint_path else None
        )
        if pinned_checkpoint_sha256 and self.checkpoint_sha256 != pinned_checkpoint_sha256:
            raise RuntimeError(
                "backbone checkpoint SHA-256 does not match the pinned value:\n"
                f"  pinned : {pinned_checkpoint_sha256}\n"
                f"  loaded : {self.checkpoint_sha256}"
            )

        self._text_features: dict[str, torch.Tensor] = {}

    # -- setup checks ----------------------------------------------------
    @staticmethod
    def _assert_normalization_matches(preprocess) -> None:
        """The projector normalizes images itself; confirm it uses the right stats."""
        normalize = None
        for transform in getattr(preprocess, "transforms", []):
            if hasattr(transform, "mean") and hasattr(transform, "std"):
                normalize = transform
        if normalize is None:  # pragma: no cover - depends on open_clip internals
            return
        for got, want, name in (
            (tuple(float(x) for x in normalize.mean), CLIP_PIXEL_MEAN, "mean"),
            (tuple(float(x) for x in normalize.std), CLIP_PIXEL_STD, "std"),
        ):
            if max(abs(a - b) for a, b in zip(got, want)) > 1e-6:
                raise RuntimeError(
                    f"pixel {name} mismatch: projector uses {want}, open_clip expects {got}"
                )

    @staticmethod
    def _resolve_checkpoint(open_clip, arch: str, pretrained: str, cache_dir) -> str | None:
        """Best-effort path to the downloaded weights, for checksum pinning."""
        try:
            url = open_clip.pretrained.get_pretrained_url(arch, pretrained)
            if url:
                path = open_clip.pretrained.download_pretrained_from_url(
                    url, cache_dir=str(cache_dir) if cache_dir else None
                )
                return str(path)
        except (AttributeError, TypeError, OSError):
            pass
        # HF-hub weights: locate the cached blob if it is already present.
        root = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface"
        if root.exists():
            candidates = sorted(root.rglob("open_clip_pytorch_model.bin")) + sorted(
                root.rglob("open_clip_model.safetensors")
            )
            if candidates:
                return str(candidates[0])
        return None

    # -- encoders --------------------------------------------------------
    def _autocast(self):
        if self.device.type == "cuda" and self.precision == "fp16":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    @torch.no_grad()
    def text_features(self, mode: str = "ensemble") -> torch.Tensor:
        """``(num_classes, embed_dim)`` L2-normalized zero-shot classifier."""
        if mode not in self._text_features:
            per_class = []
            for prompts in build_prompts(mode):
                tokens = self.tokenizer(list(prompts)).to(self.device)
                with self._autocast():
                    feats = self.model.encode_text(tokens)
                feats = feats.float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
                mean = feats.mean(dim=0)
                per_class.append(mean / mean.norm())
            self._text_features[mode] = torch.stack(per_class, dim=0)
        return self._text_features[mode]

    @torch.no_grad()
    def encode_views(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, V, 3, H, W)`` normalized depth views.

        Returns ``(B, V, embed_dim)`` L2-normalized float32 image embeddings.
        Embeddings, not logits, are the cached primitive: the prompt ablation
        rescoring then costs nothing.
        """
        if images.dim() != 5:
            raise ValueError(f"expected (B, V, 3, H, W), got {tuple(images.shape)}")
        batch, views = images.shape[:2]
        flat = images.reshape(batch * views, *images.shape[2:]).to(self.device)
        with self._autocast():
            feats = self.model.encode_image(flat)
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.reshape(batch, views, -1)

    @torch.no_grad()
    def logits_from_embeddings(
        self, embeddings: torch.Tensor, mode: str = "ensemble"
    ) -> torch.Tensor:
        """``(B, V, num_classes)`` per-view logits, scaled by the model's temperature."""
        text = self.text_features(mode).to(embeddings.device)
        scale = self.logit_scale
        return scale * embeddings @ text.t()

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp().detach().cpu())

    @property
    def embed_dim(self) -> int:
        return int(self.text_features("canonical").shape[-1])

    def info(self) -> BackboneInfo:
        return BackboneInfo(
            arch=self.arch,
            pretrained=self.pretrained,
            device=str(self.device),
            precision=self.precision,
            embed_dim=self.embed_dim,
            logit_scale=self.logit_scale,
            num_params=sum(p.numel() for p in self.model.parameters()),
            checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=self.checkpoint_sha256,
            prompt_fingerprints={m: prompt_fingerprint(m) for m in ("ensemble", "canonical")},
        )

    def assert_frozen(self) -> None:
        """Fail loudly if anything in the backbone became trainable."""
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if trainable:
            raise RuntimeError(f"backbone is not frozen: {len(trainable)} trainable tensors")
        if self.model.training:
            raise RuntimeError("backbone is in training mode")
