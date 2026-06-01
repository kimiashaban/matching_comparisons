"""
GPU memory management utilities for processing 4K images safely.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager

import torch


def get_gpu_free_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 ** 2)


def get_optimal_batch_size(
    image_height: int,
    image_width: int,
    base_batch: int = 4,
    max_pixel_budget: int = 4096 * 4096 * 8,
) -> int:
    """Heuristic batch size: scale down based on image resolution."""
    pixels_per_image = image_height * image_width
    batch = max(1, max_pixel_budget // pixels_per_image)
    return min(batch, base_batch)


def clear_gpu_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@contextmanager
def DeviceContext(device: torch.device):
    """Context manager that clears GPU cache on exit."""
    try:
        yield device
    finally:
        if device.type == "cuda":
            clear_gpu_cache()
