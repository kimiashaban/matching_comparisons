"""
LIQE: Learned Image Quality Evaluator (CVPR 2023).

Reference-free. Does NOT require text prompts.

Evaluates quality at NATIVE resolution by extracting 224x224 patches from the
full-resolution image (via unfold, no downscaling) and scoring each patch
through CLIP with quality/scene/distortion-aware text prompts. The final
score is averaged across patches. Handles any resolution >= 224px.

Memory is low even for 6K images since patches are processed individually.
Score range: ~1–5 (higher is better).

Requires: pyiqa (pip install pyiqa), opencv module on ComputeCanada.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .base import ReferenceFreeMetric, MetricResult


class LIQEMetric(ReferenceFreeMetric):
    """Native-resolution quality assessment via LIQE (CLIP patch scoring)."""

    name = "liqe"
    requires_prompts = False

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(device=device, dtype=dtype)

    def load_model(self) -> None:
        if self._model is not None:
            return
        import pyiqa

        self._model = pyiqa.create_metric("liqe", device=self.device)

    @torch.no_grad()
    def compute(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
        batch_size: int = 1,
    ) -> MetricResult:
        self.load_model()
        per_image_scores: list[float] = []

        print(f"  [liqe] Scoring {len(images)} images at native resolution "
              f"(CLIP patch-based quality)...")

        for img in images:
            tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)
            score = self._model(tensor)
            per_image_scores.append(score.item())
            del tensor, score

        mean_score = sum(per_image_scores) / len(per_image_scores)
        return MetricResult(
            name=self.name,
            value=mean_score,
            per_image=per_image_scores,
            extra={
                "num_images": len(images),
                "scale": "~1-5 (higher is better)",
            },
        )
