"""
Abstract base classes for evaluation metrics.

Three tiers:
  - Metric:              common interface (name, device management, results)
  - ReferenceFreeMetric: needs only generated images (+ optional prompts)
  - ReferenceBasedMetric: needs generated images AND a reference set
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image


@dataclass
class MetricResult:
    """Container for a single metric's output."""
    name: str
    value: float
    per_image: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        out = {"metric": self.name, "score": round(self.value, 6)}
        if self.extra:
            out["extra"] = {k: round(v, 6) if isinstance(v, float) else v
                           for k, v in self.extra.items()}
        return out


class Metric(abc.ABC):
    """Base class for all metrics."""

    name: str = "base"
    requires_prompts: bool = False
    requires_reference: bool = False

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._model = None

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load model weights onto self.device. Idempotent."""

    @abc.abstractmethod
    def compute(self, **kwargs) -> MetricResult:
        """Compute the metric. Subclasses define the signature."""

    def unload_model(self) -> None:
        """Free GPU memory."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"


class ReferenceFreeMetric(Metric):
    """Metric that only needs generated images (+ optional text prompts)."""

    requires_reference = False

    @abc.abstractmethod
    def compute(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
    ) -> MetricResult:
        ...


class ReferenceBasedMetric(Metric):
    """Metric that compares generated images against a reference set."""

    requires_reference = True

    @abc.abstractmethod
    def compute(
        self,
        images: list[Image.Image],
        reference_images: list[Image.Image],
        prompts: list[str] | None = None,
    ) -> MetricResult:
        ...
