"""
FID (Fréchet Inception Distance): measures distributional distance
between generated and reference image sets.

Reference-based. Does NOT require text prompts.
Uses Inception v3 features (2048-dim) resized to 299x299.

Note: For 4K evaluation, standard FID loses high-frequency detail
due to aggressive downsampling. Use Patch-FID for complementary signal.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy import linalg

from .base import ReferenceBasedMetric, MetricResult
from .inception_features import get_inception_model, extract_features, compute_statistics


class FIDMetric(ReferenceBasedMetric):
    name = "fid"
    requires_prompts = False

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        feature_batch_size: int = 16,
    ):
        super().__init__(device=device, dtype=dtype)
        self.feature_batch_size = feature_batch_size

    def load_model(self) -> None:
        if self._model is not None:
            return
        self._model = get_inception_model(self.device)

    @staticmethod
    def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                          mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """Compute the Fréchet distance between two multivariate Gaussians."""
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError("Imaginary component in sqrtm result")
            covmean = covmean.real

        return float(
            diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        )

    @torch.no_grad()
    def compute(
        self,
        images: list[Image.Image],
        reference_images: list[Image.Image],
        prompts: list[str] | None = None,
        batch_size: int | None = None,
    ) -> MetricResult:
        self.load_model()
        bs = batch_size or self.feature_batch_size

        print(f"  [FID] Extracting features from {len(images)} generated images...")
        gen_feats = extract_features(images, self._model, self.device, batch_size=bs)

        print(f"  [FID] Extracting features from {len(reference_images)} reference images...")
        ref_feats = extract_features(reference_images, self._model, self.device, batch_size=bs)

        mu_gen, sigma_gen = compute_statistics(gen_feats)
        mu_ref, sigma_ref = compute_statistics(ref_feats)

        fid_value = self._frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)

        return MetricResult(
            name=self.name,
            value=fid_value,
            extra={
                "num_generated": len(images),
                "num_reference": len(reference_images),
                "lower_is_better": True,
            },
        )
