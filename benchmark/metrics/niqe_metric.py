"""
NIQE: Natural Image Quality Evaluator.

Reference-free. Does NOT require text prompts.

Evaluates image quality at NATIVE resolution using natural scene statistics —
no downscaling whatsoever. Divides the image into 96x96 blocks and measures
deviation from pristine natural image statistics. Lower scores indicate
higher quality. Memory footprint is < 1 GB even for 6K images.

Note: NIQE measures "naturalness" rather than learned perceptual quality.
It may rate sharp AI-generated images lower if their statistics differ from
natural photographs.

Requires: pyiqa (pip install pyiqa), opencv module on ComputeCanada.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .base import ReferenceFreeMetric, MetricResult


class NIQEMetric(ReferenceFreeMetric):
    """Native-resolution quality assessment via NIQE."""

    name = "niqe"
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

        self._model = pyiqa.create_metric("niqe", device=self.device)

    @torch.no_grad()
    def compute(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
        batch_size: int = 1,
    ) -> MetricResult:
        self.load_model()
        per_image_scores: list[float] = []

        print(f"  [niqe] Scoring {len(images)} images at native resolution...")

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
                "scale": "lower is better",
            },
        )
