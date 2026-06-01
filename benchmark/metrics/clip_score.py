"""
CLIP Score: cosine similarity between CLIP text and image embeddings.

Reference-free. Requires text prompts.
Uses OpenAI CLIP ViT-L/14 (the standard for T2I evaluation).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

from .base import ReferenceFreeMetric, MetricResult


class CLIPScoreMetric(ReferenceFreeMetric):
    name = "clip_score"
    requires_prompts = True

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(device=device, dtype=dtype)
        self.model_name = model_name

    def load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        self._processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=True)
        self._model = CLIPModel.from_pretrained(
            self.model_name, dtype=self.dtype
        ).to(self.device).eval()

    @torch.no_grad()
    def compute(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
        batch_size: int = 8,
    ) -> MetricResult:
        if prompts is None or len(prompts) == 0:
            raise ValueError("CLIPScore requires text prompts.")
        assert len(images) == len(prompts), (
            f"Mismatch: {len(images)} images vs {len(prompts)} prompts"
        )

        self.load_model()
        per_image_scores = []

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            batch_txts = prompts[i : i + batch_size]

            inputs = self._processor(
                text=batch_txts,
                images=batch_imgs,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self._model(**inputs)
            img_emb = F.normalize(outputs.image_embeds, dim=-1)
            txt_emb = F.normalize(outputs.text_embeds, dim=-1)

            # Per-pair cosine similarity (not cross-similarity)
            scores = (img_emb * txt_emb).sum(dim=-1)
            # Standard CLIP Score scales by 100
            per_image_scores.extend((scores * 100).cpu().tolist())

            del inputs, outputs, img_emb, txt_emb, scores

        mean_score = sum(per_image_scores) / len(per_image_scores)
        return MetricResult(
            name=self.name,
            value=mean_score,
            per_image=per_image_scores,
            extra={"num_images": len(images)},
        )
