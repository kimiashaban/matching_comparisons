"""
LAION Aesthetic Score v2: predicts aesthetic quality on a 1-10 scale.

Reference-free. Does NOT require text prompts.
Uses a linear head on top of CLIP ViT-L/14 image embeddings, trained on
the LAION-Aesthetics dataset human ratings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base import ReferenceFreeMetric, MetricResult


class AestheticMLP(nn.Module):
    """The lightweight MLP head from LAION aesthetic predictor v2."""

    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AestheticScoreMetric(ReferenceFreeMetric):
    name = "aesthetic"
    requires_prompts = False

    AESTHETIC_MODEL_URL = (
        "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
        "raw/main/sac+logos+ava1-l14-linearMSE.pth"
    )

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(device=device, dtype=dtype)
        self.clip_model_name = clip_model_name

    def load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        import urllib.request
        from pathlib import Path

        self._processor = CLIPProcessor.from_pretrained(self.clip_model_name, use_fast=True)
        self._clip = CLIPModel.from_pretrained(
            self.clip_model_name, dtype=self.dtype
        ).to(self.device).eval()

        cache_dir = Path.home() / ".cache" / "aesthetic_predictor"
        cache_dir.mkdir(parents=True, exist_ok=True)
        weights_path = cache_dir / "sac+logos+ava1-l14-linearMSE.pth"
        if not weights_path.exists():
            print(f"[aesthetic] Downloading aesthetic predictor weights to {weights_path}")
            urllib.request.urlretrieve(self.AESTHETIC_MODEL_URL, str(weights_path))

        self._model = AestheticMLP(input_size=768)
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model = self._model.to(self.device).to(torch.float32).eval()

    @torch.no_grad()
    def compute(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
        batch_size: int = 16,
    ) -> MetricResult:
        self.load_model()
        per_image_scores = []

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            inputs = self._processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            img_emb = self._clip.get_image_features(**inputs)
            if not isinstance(img_emb, torch.Tensor):
                img_emb = img_emb.pooler_output
            img_emb = F.normalize(img_emb, dim=-1).to(torch.float32)

            scores = self._model(img_emb).squeeze(-1)
            per_image_scores.extend(scores.cpu().tolist())

            del inputs, img_emb, scores

        mean_score = sum(per_image_scores) / len(per_image_scores)
        return MetricResult(
            name=self.name,
            value=mean_score,
            per_image=per_image_scores,
            extra={"num_images": len(images), "scale": "1-10"},
        )
