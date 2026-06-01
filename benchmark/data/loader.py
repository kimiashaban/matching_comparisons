"""
Data loading utilities for the evaluation benchmark.

Handles:
  - Loading prompts from metadata.jsonl
  - Image folder datasets with lazy loading (critical for 4K)
  - Paired image+prompt datasets
  - Patch extraction datasets for Patch-FID
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def load_prompts_from_jsonl(path: Path | str) -> list[dict]:
    """Load prompts from a JSONL file. Each line: {"text": "...", ...}."""
    path = Path(path)
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _extract_index_from_filename(filename: str) -> int | None:
    """Extract the 5-digit index prefix from filenames like 00042_slug.png."""
    m = re.match(r"^(\d{5})_", filename)
    return int(m.group(1)) if m else None


def discover_images(directory: Path) -> list[Path]:
    """Return sorted list of image files in a directory."""
    if not directory.exists():
        return []
    imgs = [
        p for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return imgs


def pair_images_with_prompts(
    image_dir: Path,
    prompts_path: Path,
) -> list[tuple[Path, str]]:
    """
    Match images to prompts by extracting the numeric index from filenames
    (e.g., 00042_slug.png -> index 42 -> prompts[42]["text"]).
    """
    entries = load_prompts_from_jsonl(prompts_path)
    images = discover_images(image_dir)
    pairs = []
    for img_path in images:
        idx = _extract_index_from_filename(img_path.name)
        if idx is not None and idx < len(entries):
            pairs.append((img_path, entries[idx]["text"]))
    return pairs


class ImageFolderDataset(Dataset):
    """
    Lazily loads images from a folder. Each __getitem__ returns a PIL Image.
    No resizing is applied; that is the metric's responsibility.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        max_images: int | None = None,
    ):
        self.root = Path(root)
        self.paths = discover_images(self.root)
        if max_images is not None:
            self.paths = self.paths[:max_images]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Image.Image:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


class ImagePromptDataset(Dataset):
    """
    Dataset yielding (PIL Image, prompt_text) pairs.
    Images are lazily loaded to avoid OOM with 4K images.
    """

    def __init__(
        self,
        image_dir: Path | str,
        prompts_path: Path | str,
        transform: Callable | None = None,
        max_images: int | None = None,
    ):
        self.pairs = pair_images_with_prompts(Path(image_dir), Path(prompts_path))
        if max_images is not None:
            self.pairs = self.pairs[:max_images]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        img_path, prompt = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, prompt


class PatchDataset(Dataset):
    """
    Extracts random square patches from images at native resolution.
    Used for Patch-FID computation to preserve high-res local detail.
    """

    def __init__(
        self,
        root: Path | str,
        patch_size: int = 299,
        patches_per_image: int = 8,
        seed: int = 42,
        max_images: int | None = None,
    ):
        self.root = Path(root)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.paths = discover_images(self.root)
        if max_images is not None:
            self.paths = self.paths[:max_images]
        self.rng = random.Random(seed)
        self._precompute_patch_coords()

    def _precompute_patch_coords(self) -> None:
        """Pre-sample patch positions so results are deterministic."""
        self.patch_specs: list[tuple[int, int, int]] = []  # (img_idx, y, x)
        for img_idx, img_path in enumerate(self.paths):
            img = Image.open(img_path)
            w, h = img.size
            img.close()
            max_y = max(0, h - self.patch_size)
            max_x = max(0, w - self.patch_size)
            for _ in range(self.patches_per_image):
                y = self.rng.randint(0, max_y) if max_y > 0 else 0
                x = self.rng.randint(0, max_x) if max_x > 0 else 0
                self.patch_specs.append((img_idx, y, x))

    def __len__(self) -> int:
        return len(self.patch_specs)

    def __getitem__(self, idx: int) -> Image.Image:
        img_idx, y, x = self.patch_specs[idx]
        img = Image.open(self.paths[img_idx]).convert("RGB")
        patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
        img.close()
        return patch


def make_batched_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 2,
    collate_fn: Callable | None = None,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for large-image evaluation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        prefetch_factor=2 if num_workers > 0 else None,
    )
