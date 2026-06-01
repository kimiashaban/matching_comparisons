"""
Patch-FID: FID computed on randomly cropped patches at native resolution.

Reference-based. Does NOT require text prompts.

Instead of resizing the full 4K image to 299x299 (which destroys detail),
this metric extracts random 299x299 patches from the native-resolution
image and computes FID over those patches. This captures local texture
and detail quality that standard FID misses entirely.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from scipy import linalg

from .base import ReferenceBasedMetric, MetricResult
from .inception_features import (
    get_inception_model,
    extract_features,
    compute_statistics,
    INCEPTION_INPUT_SIZE,
)


def _extract_patches(
    images: list[Image.Image],
    patch_size: int = INCEPTION_INPUT_SIZE,
    patches_per_image: int = 8,
    seed: int = 42,
) -> list[Image.Image]:
    """Extract random square patches from a list of PIL images."""
    rng = random.Random(seed)
    patches = []
    for img in images:
        w, h = img.size
        max_y = max(0, h - patch_size)
        max_x = max(0, w - patch_size)
        for _ in range(patches_per_image):
            y = rng.randint(0, max_y) if max_y > 0 else 0
            x = rng.randint(0, max_x) if max_x > 0 else 0
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
    return patches


class PatchFIDMetric(ReferenceBasedMetric):
    name = "patch_fid"
    requires_prompts = False

    def __init__(
        self,
        patch_size: int = INCEPTION_INPUT_SIZE,
        patches_per_image: int = 8,
        seed: int = 42,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        feature_batch_size: int = 32,
    ):
        super().__init__(device=device, dtype=dtype)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.seed = seed
        self.feature_batch_size = feature_batch_size

    def load_model(self) -> None:
        if self._model is not None:
            return
        self._model = get_inception_model(self.device)

    @staticmethod
    def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                          mu2: np.ndarray, sigma2: np.ndarray) -> float:
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
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

        print(f"  [Patch-FID] Extracting {self.patches_per_image} patches/image "
              f"from {len(images)} generated images ({self.patch_size}x{self.patch_size})...")
        gen_patches = _extract_patches(
            images, self.patch_size, self.patches_per_image, self.seed
        )

        print(f"  [Patch-FID] Extracting {self.patches_per_image} patches/image "
              f"from {len(reference_images)} reference images...")
        ref_patches = _extract_patches(
            reference_images, self.patch_size, self.patches_per_image, self.seed + 1
        )

        from torchvision import transforms
        patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        gen_feats = extract_features(gen_patches, self._model, self.device, bs, patch_transform)
        ref_feats = extract_features(ref_patches, self._model, self.device, bs, patch_transform)

        mu_gen, sigma_gen = compute_statistics(gen_feats)
        mu_ref, sigma_ref = compute_statistics(ref_feats)

        pfid = self._frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)

        return MetricResult(
            name=self.name,
            value=pfid,
            extra={
                "num_gen_patches": len(gen_patches),
                "num_ref_patches": len(ref_patches),
                "patch_size": self.patch_size,
                "patches_per_image": self.patches_per_image,
                "lower_is_better": True,
            },
        )
