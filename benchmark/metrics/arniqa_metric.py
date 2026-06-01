"""
ARNIQA: Learning Distortion Manifold for Image Quality Assessment (WACV 2024).

Reference-free. Does NOT require text prompts.

Fully convolutional ResNet50 backbone — no fixed input size, processes
images at native resolution. Also creates a 0.5x downscaled copy for
multi-scale feature extraction. Trained via contrastive learning on
distortion manifolds. Handles any resolution.

Score range: ~0–1 (higher is better).

Requires: pyiqa (pip install pyiqa), opencv module on ComputeCanada.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .base import ReferenceFreeMetric, MetricResult


class ARNIQAMetric(ReferenceFreeMetric):
    """Native-resolution quality assessment via ARNIQA."""

    name = "arniqa"
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

        self._model = pyiqa.create_metric("arniqa", device=self.device)

    @torch.no_grad()
    def compute(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
        batch_size: int = 1,
    ) -> MetricResult:
        self.load_model()
        per_image_scores: list[float] = []

        print(f"  [arniqa] Scoring {len(images)} images at native resolution "
              f"(multi-scale contrastive features)...")

        for img in images:
            tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)
            score = self._model(tensor)
            per_image_scores.append(score.item())
            del tensor, score
            torch.cuda.empty_cache()

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
