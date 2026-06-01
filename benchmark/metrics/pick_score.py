"""
PickScore: CLIP-H–based human-preference scoring for text-to-image pairs.

Reference-free. **Requires** text prompts (one per image).

PickScore is a CLIP ViT-H/14 model fine-tuned on the Pick-a-Pic preference
dataset.  For each (prompt, image) pair it returns a scalar proportional to
the cosine similarity between L2-normalized text and image embeddings,
scaled by the learned ``logit_scale`` (same convention as CLIP).

Higher scores indicate better alignment with human preferences on the
Pick-a-Pic distribution.

References
----------
- Kirstain et al., "Pick-a-Pic: An Open Dataset of User Preferences for
  Text-to-Image Generation", ICML 2023.
- Model card: https://huggingface.co/yuvalkirstain/PickScore_v1

Dependencies: ``transformers`` (already in benchmark requirements).
"""

from __future__ import annotations

import inspect

import torch
from PIL import Image

from .base import ReferenceFreeMetric, MetricResult


class PickScoreMetric(ReferenceFreeMetric):
    name = "pick_score"
    requires_prompts = True

    def __init__(
        self,
        processor_name_or_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_name_or_path: str = "yuvalkirstain/PickScore_v1",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(device=device, dtype=dtype)
        self.processor_name_or_path = processor_name_or_path
        self.model_name_or_path = model_name_or_path
        self._processor = None

    def load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "PickScore requires Hugging Face ``transformers``.\n"
                "Install: pip install transformers"
            ) from e

        self._processor = AutoProcessor.from_pretrained(self.processor_name_or_path)
        load_kw: dict = {}
        if "dtype" in inspect.signature(AutoModel.from_pretrained).parameters:
            load_kw["dtype"] = self.dtype
        else:
            load_kw["torch_dtype"] = self.dtype
        self._model = AutoModel.from_pretrained(
            self.model_name_or_path,
            **load_kw,
        ).to(self.device).eval()

    def unload_model(self) -> None:
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def compute(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
        batch_size: int = 4,
    ) -> MetricResult:
        if prompts is None or len(prompts) == 0:
            raise ValueError("PickScore requires text prompts.")
        if len(images) != len(prompts):
            raise ValueError(
                f"PickScore: {len(images)} images vs {len(prompts)} prompts"
            )

        self.load_model()
        assert self._processor is not None

        per_image_scores: list[float] = []
        logit_scale = self._model.logit_scale.exp()

        for i in range(0, len(images), batch_size):
            batch_imgs = [im.convert("RGB") for im in images[i: i + batch_size]]
            batch_txts = prompts[i: i + batch_size]

            image_inputs = self._processor(
                images=batch_imgs,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            text_inputs = self._processor(
                text=batch_txts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )

            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

            image_embs = self._model.get_image_features(**image_inputs)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

            text_embs = self._model.get_text_features(**text_inputs)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

            # Per-pair score (diagonal of text @ image.T), matches HF model card.
            sims = (text_embs * image_embs).sum(dim=-1)
            scores = (logit_scale * sims).float().cpu().tolist()
            if isinstance(scores, float):
                per_image_scores.append(scores)
            else:
                per_image_scores.extend(float(s) for s in scores)

            del image_inputs, text_inputs, image_embs, text_embs, sims

        mean_score = sum(per_image_scores) / len(per_image_scores)
        return MetricResult(
            name=self.name,
            value=mean_score,
            per_image=per_image_scores,
            extra={
                "num_images": len(images),
                "higher_is_better": True,
                "processor": self.processor_name_or_path,
                "weights": self.model_name_or_path,
            },
        )
