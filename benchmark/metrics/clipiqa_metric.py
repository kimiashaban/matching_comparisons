"""
CLIP-IQA+: CLIP-based Image Quality Assessment.

Reference-free. Does NOT require text prompts.

Uses CLIP ViT-L/14 with learned quality-aware prompts to evaluate image
quality. Unlike the aesthetic predictor (which measures "is this beautiful?"),
CLIP-IQA+ is trained specifically for technical quality assessment
(sharpness, artifacts, noise). Images are resized to 512px internally.
Score range: 0–1 (higher is better).

Requires: pyiqa (pip install pyiqa), opencv module on ComputeCanada.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .base import ReferenceFreeMetric, MetricResult


class CLIPIQAMetric(ReferenceFreeMetric):
    """Quality assessment via CLIP-IQA+ (ViT-L/14, 512px)."""

    name = "clipiqa"
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

        self._model = pyiqa.create_metric(
            "clipiqa+_vitL14_512", device=self.device
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

        print(f"  [clipiqa] Scoring {len(images)} images (CLIP-IQA+ ViT-L/14)...")

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            batch_tensor = torch.stack([
                TF.to_tensor(img.resize((512, 512), Image.LANCZOS))
                for img in batch_imgs
            ]).to(self.device)

            scores = self._model(batch_tensor)
            per_image_scores.extend(scores.squeeze(-1).cpu().tolist())

            del batch_tensor, scores

        mean_score = sum(per_image_scores) / len(per_image_scores)
        return MetricResult(
            name=self.name,
            value=mean_score,
            per_image=per_image_scores,
            extra={
                "num_images": len(images),
                "scale": "0-1 (higher is better)",
            },
        )
