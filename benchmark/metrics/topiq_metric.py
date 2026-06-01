"""
TOPIQ-NR: Top-down Image Quality assessment (whole-image, no-reference).

Reference-free. Does NOT require text prompts.

State-of-the-art NR-IQA metric (IEEE TIP 2024, SRCC=0.93 on KonIQ-10k).
Evaluates whole-image technical quality — sharpness, detail, artifacts,
texture degradation — using cross-scale attention on a ResNet50 backbone.
Images are resized to 512px (longest side) before scoring, matching the
resolution range the model was trained on. Scores on a 0-1 scale
(higher is better).

Requires: pyiqa (pip install pyiqa), opencv module on ComputeCanada.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .base import ReferenceFreeMetric, MetricResult


INPUT_SIZE = 512


class TOPIQMetric(ReferenceFreeMetric):
    """Whole-image quality assessment via TOPIQ-NR."""

    name = "topiq_nr"
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

        self._model = pyiqa.create_metric("topiq_nr", device=self.device)

    @staticmethod
    def _resize(img: Image.Image, max_side: int = INPUT_SIZE) -> Image.Image:
        w, h = img.size
        if max(w, h) <= max_side:
            return img
        scale = max_side / max(w, h)
        return img.resize(
            (round(w * scale), round(h * scale)), Image.LANCZOS
        )

    @torch.no_grad()
    def compute(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
        batch_size: int = 8,
    ) -> MetricResult:
        self.load_model()
        per_image_scores: list[float] = []

        print(f"  [topiq_nr] Scoring {len(images)} images "
              f"(resized to {INPUT_SIZE}px longest side)...")

        # Score one at a time since resized images may have different sizes
        for img in images:
            resized = self._resize(img)
            tensor = TF.to_tensor(resized).unsqueeze(0).to(self.device)

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
                "input_size": INPUT_SIZE,
                "scale": "0-1 (higher is better)",
            },
        )
