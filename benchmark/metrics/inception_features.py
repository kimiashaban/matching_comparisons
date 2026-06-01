"""
Shared Inception v3 feature extraction for FID and KID metrics.

Extracts the 2048-dim features from the penultimate pooling layer,
which is the standard for FID/KID computation.
Images are resized to 299x299 (Inception's native input size).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.memory import clear_gpu_cache


INCEPTION_INPUT_SIZE = 299

inception_transform = transforms.Compose([
    transforms.Resize((INCEPTION_INPUT_SIZE, INCEPTION_INPUT_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_inception_model(device: torch.device) -> torch.nn.Module:
    """Load Inception v3 in eval mode with the classification head removed."""
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_features(
    images: list[Image.Image],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
    transform: transforms.Compose | None = None,
) -> np.ndarray:
    """
    Extract 2048-dim Inception features from a list of PIL images.
    Returns a (N, 2048) numpy array.
    """
    if transform is None:
        transform = inception_transform

    all_features = []
    for i in range(0, len(images), batch_size):
        batch_pil = images[i : i + batch_size]
        batch_tensors = torch.stack([transform(img) for img in batch_pil])
        batch_tensors = batch_tensors.to(device)

        features = model(batch_tensors)
        if isinstance(features, tuple):
            features = features[0]
        all_features.append(features.cpu().numpy())

        del batch_tensors, features

    clear_gpu_cache()
    return np.concatenate(all_features, axis=0)


def compute_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of feature vectors for FID."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma
