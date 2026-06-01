"""
NRQM: No-Reference Quality Metric for super-resolution evaluation.

Reference-free. Does NOT require text prompts.

Evaluates image quality at NATIVE resolution using steerable pyramid
decomposition — no downscaling whatsoever. Specifically designed for
assessing super-resolution output quality. Scores on a ~1–10 scale
(higher is better). Memory footprint is ~2–4 GB even for 6K images.

Requires: pyiqa (pip install pyiqa), opencv module on ComputeCanada.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .base import ReferenceFreeMetric, MetricResult


class NRQMMetric(ReferenceFreeMetric):
    """Native-resolution quality assessment via NRQM."""

    name = "nrqm"
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

        self._model = pyiqa.create_metric("nrqm", device=self.device)

    @torch.no_grad()
    def compute(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
        batch_size: int = 1,
    ) -> MetricResult:
        self.load_model()
        per_image_scores: list[float] = []

        print(f"  [nrqm] Scoring {len(images)} images at native resolution...")

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
                "scale": "~1-10 (higher is better)",
            },
        )
