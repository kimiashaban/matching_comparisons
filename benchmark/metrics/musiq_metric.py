"""
MUSIQ: Multi-Scale Image Quality Transformer (ICCV 2021, Google).

Reference-free. Does NOT require text prompts.

Evaluates image quality at NATIVE resolution — no downscaling. Uses
multi-scale patch tokenization: extracts 32x32 patches from the original
resolution plus two downscaled copies (224px, 384px longest side), and
processes all patches through a 14-layer Transformer with hash-based
spatial embeddings. This captures both global structure and local detail.

For a 6144x6144 image: ~37K tokens, ~17 GB peak memory (fits on H100 80GB).
Score range: ~0–100 (higher is better).

Requires: pyiqa (pip install pyiqa), opencv module on ComputeCanada.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .base import ReferenceFreeMetric, MetricResult


class MUSIQMetric(ReferenceFreeMetric):
    """Native-resolution quality assessment via MUSIQ."""

    name = "musiq"
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

        self._model = pyiqa.create_metric("musiq", device=self.device)

    @torch.no_grad()
    def compute(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
        batch_size: int = 1,
    ) -> MetricResult:
        self.load_model()
        per_image_scores: list[float] = []

        print(f"  [musiq] Scoring {len(images)} images at native resolution "
              f"(multi-scale patch tokenization)...")

        # Process one at a time — large images produce many tokens
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
                "scale": "~0-100 (higher is better)",
            },
        )
