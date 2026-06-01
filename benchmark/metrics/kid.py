"""
KID (Kernel Inception Distance): unbiased MMD^2 estimator between
generated and reference feature distributions.

Reference-based. Does NOT require text prompts.
Uses Inception v3 features like FID, but the kernel-based estimator
is unbiased and more reliable at small sample sizes.
Lower is better.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .base import ReferenceBasedMetric, MetricResult
from .inception_features import get_inception_model, extract_features


def _polynomial_mmd_unbiased(
    X: np.ndarray,
    Y: np.ndarray,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
    num_subsets: int = 100,
    subset_size: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute unbiased polynomial kernel MMD^2 with bootstrap variance.
    Returns (mean_kid, std_kid) over random subsets.
    """
    rng = np.random.RandomState(seed)
    n = min(len(X), len(Y))
    subset_size = min(subset_size, n)

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    def _kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return (gamma * (A @ B.T) + coef0) ** degree

    kid_values = []
    for _ in range(num_subsets):
        idx_x = rng.choice(len(X), subset_size, replace=False)
        idx_y = rng.choice(len(Y), subset_size, replace=False)
        Xs, Ys = X[idx_x], Y[idx_y]

        Kxx = _kernel(Xs, Xs)
        Kyy = _kernel(Ys, Ys)
        Kxy = _kernel(Xs, Ys)

        m = subset_size
        # Unbiased estimator: exclude diagonal
        mmd2 = (
            (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
            - 2.0 * Kxy.mean()
        )
        kid_values.append(float(mmd2))

    return float(np.mean(kid_values)), float(np.std(kid_values))


class KIDMetric(ReferenceBasedMetric):
    name = "kid"
    requires_prompts = False

    def __init__(
        self,
        num_subsets: int = 100,
        subset_size: int = 1000,
        degree: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        feature_batch_size: int = 16,
    ):
        super().__init__(device=device, dtype=dtype)
        self.num_subsets = num_subsets
        self.subset_size = subset_size
        self.degree = degree
        self.feature_batch_size = feature_batch_size

    def load_model(self) -> None:
        if self._model is not None:
            return
        self._model = get_inception_model(self.device)

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

        print(f"  [KID] Extracting features from {len(images)} generated images...")
        gen_feats = extract_features(images, self._model, self.device, batch_size=bs)

        print(f"  [KID] Extracting features from {len(reference_images)} reference images...")
        ref_feats = extract_features(reference_images, self._model, self.device, batch_size=bs)

        kid_mean, kid_std = _polynomial_mmd_unbiased(
            gen_feats, ref_feats,
            degree=self.degree,
            num_subsets=self.num_subsets,
            subset_size=min(self.subset_size, min(len(gen_feats), len(ref_feats))),
        )

        return MetricResult(
            name=self.name,
            value=kid_mean,
            extra={
                "kid_std": kid_std,
                "num_generated": len(images),
                "num_reference": len(reference_images),
                "num_subsets": self.num_subsets,
                "lower_is_better": True,
            },
        )
