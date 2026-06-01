"""
ImageReward: a learned reward model for text-to-image generation.

Reference-free. Requires text prompts.
Trained on human preference rankings; outputs a scalar reward.
Higher is better.

Note: The installed `image-reward` package may need a one-time patch
on Compute Canada clusters. See README for details.
"""

from __future__ import annotations

import torch
from PIL import Image

from .base import ReferenceFreeMetric, MetricResult


class ImageRewardMetric(ReferenceFreeMetric):
    name = "image_reward"
    requires_prompts = True

    def __init__(
        self,
        model_name: str = "ImageReward-v1.0",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(device=device, dtype=dtype)
        self.model_name = model_name

    def load_model(self) -> None:
        if self._model is not None:
            return
        import ImageReward as ir_module
        self._model = ir_module.load(self.model_name, device=self.device)

    @torch.no_grad()
    def compute(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
        batch_size: int = 4,
    ) -> MetricResult:
        if prompts is None or len(prompts) == 0:
            raise ValueError("ImageReward requires text prompts.")
        assert len(images) == len(prompts), (
            f"Mismatch: {len(images)} images vs {len(prompts)} prompts"
        )

        self.load_model()
        per_image_scores = []

        for img, txt in zip(images, prompts):
            score = self._model.score(txt, img)
            per_image_scores.append(float(score))

        mean_score = sum(per_image_scores) / len(per_image_scores)
        return MetricResult(
            name=self.name,
            value=mean_score,
            per_image=per_image_scores,
            extra={"num_images": len(images)},
        )
