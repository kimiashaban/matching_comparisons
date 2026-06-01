"""
Human Preference Score (HPS) via the hpsv2 package: preference metrics for
text-to-image generation.

Reference-free. Requires text prompts.
Higher is better.

We expose two checkpoints:
  - hps_v2   → library default for ``score`` (HPS v2.0 weights)
  - hps_v2_1 → HPS v2.1 (pass ``hps_version="v2.1"``; scores are not comparable to v2.0)
"""

from __future__ import annotations

import torch
from PIL import Image

from .base import ReferenceFreeMetric, MetricResult


class _HPSMetricBase(ReferenceFreeMetric):
    """Shared loader and batched scoring; subclasses set ``name`` and ``hps_version``."""

    name = "hps_v2"
    requires_prompts = True
    hps_version: str = "v2.0"

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(device=device, dtype=dtype)

    def load_model(self) -> None:
        if self._model is not None:
            return
        import hpsv2

        self._model = hpsv2

    @torch.no_grad()
    def compute(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
        batch_size: int = 4,
    ) -> MetricResult:
        if prompts is None or len(prompts) == 0:
            raise ValueError(f"{self.name} requires text prompts.")
        assert len(images) == len(prompts), (
            f"Mismatch: {len(images)} images vs {len(prompts)} prompts"
        )

        self.load_model()
        per_image_scores = []

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            batch_txts = prompts[i : i + batch_size]

            for img, txt in zip(batch_imgs, batch_txts):
                score = self._model.score(
                    img, txt, hps_version=self.hps_version
                )
                per_image_scores.append(
                    float(score[0]) if hasattr(score, "__len__") else float(score)
                )

        mean_score = sum(per_image_scores) / len(per_image_scores)
        return MetricResult(
            name=self.name,
            value=mean_score,
            per_image=per_image_scores,
            extra={"num_images": len(images), "hps_version": self.hps_version},
        )


class HPSv2Metric(_HPSMetricBase):
    """HPS v2.0 checkpoint (explicit ``hps_version="v2.0"``)."""

    name = "hps_v2"
    hps_version = "v2.0"


class HPSv21Metric(_HPSMetricBase):
    """HPS v2.1 checkpoint (higher-quality training data per upstream release)."""

    name = "hps_v2_1"
    hps_version = "v2.1"
