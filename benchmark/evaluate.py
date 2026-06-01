#!/usr/bin/env python3
"""
Aesthetic-4K Evaluation Benchmark
=================================

Comprehensive evaluation pipeline for high-resolution image generation models.
Computes reference-free metrics (CLIP Score, ImageReward, Aesthetic, HPS v2 / v2.1, PickScore)
and reference-based metrics (FID, Patch-FID, KID) with careful memory
management for 4K images.

Usage Examples
--------------
  # Auto-discover by resolution + model
  python evaluate.py --resolution 4096x4096 --model SEGA-Flux --metrics clip_score aesthetic

  # Explicit path
  python evaluate.py --image-dir /path/to/images --metrics clip_score image_reward aesthetic hps_v2 hps_v2_1

  # All reference-free metrics
  python evaluate.py --resolution 4096x4096 --model SEGA-Flux --metrics all_free

  # Reference-based metrics (requires --reference-dir)
  python evaluate.py --resolution 4096x4096 --model SEGA-Flux \\
      --reference-dir /path/to/real_images --metrics fid patch_fid kid

  # Full sweep across multiple models
  python evaluate.py --resolution 4096x4096 --model SEGA-Flux SEGA-Qwen Base-Flux Base-Qwen \\
      --metrics all_free --output results.json

  # Batch all models for a resolution
  python evaluate.py --resolution 4096x4096 --model all \\
      --metrics clip_score aesthetic --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# -- local imports --------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.loader import (
    discover_images,
    load_prompts_from_jsonl,
    pair_images_with_prompts,
)
from metrics import (
    METRIC_REGISTRY,
    REFERENCE_FREE_METRICS,
    REFERENCE_BASED_METRICS,
    ReferenceFreeMetric,
    ReferenceBasedMetric,
)
from metrics.base import MetricResult
from utils.memory import clear_gpu_cache, get_optimal_batch_size

# -- constants ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES_ROOT = REPO_ROOT / "outputs_by_res"
DEFAULT_PROMPTS_PATH = REPO_ROOT / "data" / "aesthetic4k_metadata.jsonl"

ALL_RESOLUTIONS = ["2048x4096", "4096x2048", "3072x3072", "4096x4096"]


# =========================================================================
# Path discovery
# =========================================================================

def resolve_image_dir(
    image_dir: Path | None,
    resolution: str | None,
    model: str | None,
    samples_root: Path = DEFAULT_SAMPLES_ROOT,
) -> Path:
    """Resolve the image directory from explicit path or (resolution, model)."""
    if image_dir is not None:
        p = Path(image_dir)
        if not p.exists():
            raise FileNotFoundError(f"Image directory not found: {p}")
        return p

    if resolution is None or model is None:
        raise ValueError("Provide either --image-dir, or both --resolution and --model")

    p = samples_root / resolution / model
    if not p.exists():
        raise FileNotFoundError(
            f"Auto-discovered path does not exist: {p}\n"
            f"Available models at {samples_root / resolution}: "
            f"{[d.name for d in (samples_root / resolution).iterdir() if d.is_dir()] if (samples_root / resolution).exists() else 'N/A'}"
        )
    return p


def discover_all_models(resolution: str, samples_root: Path = DEFAULT_SAMPLES_ROOT) -> list[str]:
    """List all model directories under a resolution folder."""
    res_dir = samples_root / resolution
    if not res_dir.exists():
        return []
    return sorted([
        d.name for d in res_dir.iterdir()
        if d.is_dir() and not d.name.startswith((".", "a_temp"))
    ])


# =========================================================================
# Image loading with memory management
# =========================================================================

def load_images_batched(
    image_dir: Path,
    max_images: int | None = None,
) -> list[Image.Image]:
    """
    Load all images from a directory into memory.
    For 4K images this can be memory-intensive; we convert to RGB immediately
    and rely on the metric's internal batching for GPU memory.
    """
    paths = discover_images(image_dir)
    if max_images is not None:
        paths = paths[:max_images]

    images = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"  Warning: could not load {p.name}: {e}", file=sys.stderr)
    return images


def load_prompts_for_images(
    image_dir: Path,
    prompts_path: Path = DEFAULT_PROMPTS_PATH,
) -> list[str] | None:
    """Load prompts matching the images in image_dir by filename index."""
    if not prompts_path.exists():
        print(f"  Warning: prompts file not found at {prompts_path}", file=sys.stderr)
        return None
    pairs = pair_images_with_prompts(image_dir, prompts_path)
    if not pairs:
        print(f"  Warning: no image-prompt pairs found for {image_dir}", file=sys.stderr)
        return None
    return [prompt for _, prompt in pairs]


# =========================================================================
# Metric execution
# =========================================================================

def resolve_metric_names(requested: list[str]) -> list[str]:
    """Expand special aliases and validate metric names."""
    expanded = []
    for name in requested:
        if name == "all":
            expanded.extend(sorted(METRIC_REGISTRY.keys()))
        elif name == "all_free":
            expanded.extend(sorted(REFERENCE_FREE_METRICS))
        elif name == "all_ref":
            expanded.extend(sorted(REFERENCE_BASED_METRICS))
        elif name in METRIC_REGISTRY:
            expanded.append(name)
        else:
            raise ValueError(
                f"Unknown metric: '{name}'. Available: {sorted(METRIC_REGISTRY.keys())} "
                f"+ aliases: all, all_free, all_ref"
            )
    return list(dict.fromkeys(expanded))  # deduplicate preserving order


def run_single_evaluation(
    image_dir: Path,
    metric_names: list[str],
    prompts_path: Path,
    reference_dir: Path | None,
    device: str,
    batch_size: int,
    max_images: int | None,
) -> list[MetricResult]:
    """Run all requested metrics on a single image directory."""
    results = []
    images: list[Image.Image] | None = None
    prompts: list[str] | None = None
    ref_images: list[Image.Image] | None = None

    for metric_name in metric_names:
        metric_cls = METRIC_REGISTRY[metric_name]
        metric = metric_cls(device=device)

        # Lazy-load images the first time any metric needs them
        if images is None:
            print(f"  Loading images from {image_dir}...")
            images = load_images_batched(image_dir, max_images)
            print(f"  Loaded {len(images)} images.")

        if not images:
            print(f"  Skipping {metric_name}: no images found.", file=sys.stderr)
            continue

        # Load prompts if metric needs them
        if metric.requires_prompts and prompts is None:
            print(f"  Loading prompts from {prompts_path}...")
            # Re-pair because max_images might have truncated
            prompts = load_prompts_for_images(image_dir, prompts_path)
            if prompts and max_images:
                prompts = prompts[:max_images]

        if metric.requires_prompts and (prompts is None or len(prompts) != len(images)):
            n_prompts = len(prompts) if prompts else 0
            print(
                f"  Warning: {metric_name} requires prompts but found "
                f"{n_prompts} prompts for {len(images)} images. "
                f"Using min of both.",
                file=sys.stderr,
            )
            if prompts:
                n = min(len(images), len(prompts))
                images_for_metric = images[:n]
                prompts_for_metric = prompts[:n]
            else:
                print(f"  Skipping {metric_name}: no prompts available.", file=sys.stderr)
                continue
        else:
            images_for_metric = images
            prompts_for_metric = prompts

        # Load reference images for reference-based metrics
        if isinstance(metric, ReferenceBasedMetric):
            if reference_dir is None:
                print(
                    f"  Skipping {metric_name}: reference-based metric but "
                    f"--reference-dir not provided.",
                    file=sys.stderr,
                )
                continue
            if ref_images is None:
                print(f"  Loading reference images from {reference_dir}...")
                ref_images = load_images_batched(Path(reference_dir), max_images)
                print(f"  Loaded {len(ref_images)} reference images.")
            if not ref_images:
                print(f"  Skipping {metric_name}: no reference images found.", file=sys.stderr)
                continue

        # --- Execute ---
        t0 = time.time()
        print(f"  Computing {metric_name}...", flush=True)
        try:
            metric.load_model()
            if isinstance(metric, ReferenceBasedMetric):
                result = metric.compute(
                    images=images_for_metric,
                    reference_images=ref_images,
                    prompts=prompts_for_metric,
                    batch_size=batch_size,
                )
            else:
                result = metric.compute(
                    images=images_for_metric,
                    prompts=prompts_for_metric,
                    batch_size=batch_size,
                )
            elapsed = time.time() - t0
            result.extra["elapsed_sec"] = round(elapsed, 2)
            results.append(result)
            print(f"  {metric_name} = {result.value:.6f}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  ERROR computing {metric_name}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
        finally:
            metric.unload_model()
            clear_gpu_cache()

    return results


# =========================================================================
# Output formatting
# =========================================================================

def print_results_table(all_results: dict[str, list[MetricResult]]) -> None:
    """Print a clean summary table to the console."""
    if not all_results:
        print("No results to display.")
        return

    all_metric_names = []
    for results in all_results.values():
        for r in results:
            if r.name not in all_metric_names:
                all_metric_names.append(r.name)

    # Header
    col_width = max(14, max(len(m) for m in all_metric_names) + 2)
    model_width = max(20, max(len(k) for k in all_results) + 2)

    header = f"{'Model':<{model_width}}" + "".join(
        f"{m:>{col_width}}" for m in all_metric_names
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for model_key, results in all_results.items():
        scores = {r.name: r.value for r in results}
        row = f"{model_key:<{model_width}}"
        for m in all_metric_names:
            if m in scores:
                row += f"{scores[m]:>{col_width}.4f}"
            else:
                row += f"{'N/A':>{col_width}}"
        print(row)

    print("=" * len(header) + "\n")


def save_results(
    all_results: dict[str, list[MetricResult]],
    output_path: Path,
) -> None:
    """Save results to JSON or CSV based on file extension."""
    suffix = output_path.suffix.lower()

    if suffix == ".json":
        data = {}
        for model_key, results in all_results.items():
            data[model_key] = {r.name: r.summary() for r in results}
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {output_path}")

    elif suffix == ".csv":
        all_metric_names = []
        for results in all_results.values():
            for r in results:
                if r.name not in all_metric_names:
                    all_metric_names.append(r.name)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + all_metric_names)
            for model_key, results in all_results.items():
                scores = {r.name: r.value for r in results}
                row = [model_key] + [
                    f"{scores[m]:.6f}" if m in scores else ""
                    for m in all_metric_names
                ]
                writer.writerow(row)
        print(f"Results saved to {output_path}")

    else:
        # Default to JSON
        json_path = output_path.with_suffix(".json")
        save_results(all_results, json_path)


def save_per_image_results(
    all_results: dict[str, list[MetricResult]],
    output_dir: Path,
) -> None:
    """Save per-image scores to separate JSON files for detailed analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_key, results in all_results.items():
        per_image = {}
        for r in results:
            if r.per_image:
                per_image[r.name] = r.per_image
        if per_image:
            path = output_dir / f"{model_key.replace('/', '_')}_per_image.json"
            with open(path, "w") as f:
                json.dump(per_image, f, indent=2)


# =========================================================================
# CLI
# =========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aesthetic-4K Evaluation Benchmark for High-Resolution Image Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input specification ---
    input_group = parser.add_argument_group("Input (choose one mode)")
    input_group.add_argument(
        "--image-dir", type=Path, default=None,
        help="Explicit path to a directory of generated images.",
    )
    input_group.add_argument(
        "--resolution", "-r", type=str, default=None,
        choices=ALL_RESOLUTIONS,
        help="Resolution folder (e.g. 4096x4096). Used with --model for auto-discovery.",
    )
    input_group.add_argument(
        "--model", "-m", type=str, nargs="+", default=None,
        help="Model name(s) (e.g. SEGA-Flux Base-Qwen). Use 'all' to sweep all models.",
    )
    input_group.add_argument(
        "--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT,
        help=f"Root directory for auto-discovery (default: {DEFAULT_SAMPLES_ROOT}).",
    )

    # --- Metrics ---
    metric_group = parser.add_argument_group("Metrics")
    metric_group.add_argument(
        "--metrics", type=str, nargs="+", required=True,
        help=(
            f"Metrics to compute. Available: {sorted(METRIC_REGISTRY.keys())}. "
            "Aliases: 'all', 'all_free' (reference-free only), 'all_ref' (reference-based only)."
        ),
    )

    # --- Reference images (for FID/KID) ---
    ref_group = parser.add_argument_group("Reference images (for FID, Patch-FID, KID)")
    ref_group.add_argument(
        "--reference-dir", type=Path, default=None,
        help="Directory of reference/real images for distributional metrics.",
    )

    # --- Prompts ---
    prompt_group = parser.add_argument_group("Prompts")
    prompt_group.add_argument(
        "--prompts-path", type=Path, default=DEFAULT_PROMPTS_PATH,
        help=f"Path to metadata.jsonl (default: {DEFAULT_PROMPTS_PATH}).",
    )

    # --- Processing ---
    proc_group = parser.add_argument_group("Processing")
    proc_group.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for metric computation (default: 4, tune for GPU memory).",
    )
    proc_group.add_argument(
        "--max-images", type=int, default=None,
        help="Limit the number of images to evaluate (useful for debugging).",
    )
    proc_group.add_argument(
        "--device", type=str, default="cuda",
        help="PyTorch device (default: cuda).",
    )

    # --- Output ---
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Save results to this file (.json or .csv).",
    )
    out_group.add_argument(
        "--save-per-image", type=Path, default=None,
        help="Directory to save per-image metric scores.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve which metrics to compute
    metric_names = resolve_metric_names(args.metrics)
    print(f"Metrics to compute: {metric_names}")

    # Check if reference-based metrics are requested without --reference-dir
    needs_ref = any(m in REFERENCE_BASED_METRICS for m in metric_names)
    if needs_ref and args.reference_dir is None:
        print(
            "Warning: reference-based metrics requested but --reference-dir not provided. "
            "Those metrics will be skipped.\n",
            file=sys.stderr,
        )

    # Resolve model list
    if args.model and "all" in args.model:
        if args.resolution is None:
            print("Error: --model all requires --resolution", file=sys.stderr)
            return 1
        models = discover_all_models(args.resolution, args.samples_root)
        print(f"Auto-discovered models for {args.resolution}: {models}")
    elif args.model:
        models = args.model
    elif args.image_dir:
        models = [args.image_dir.name]
    else:
        print("Error: provide --image-dir, or --resolution + --model", file=sys.stderr)
        return 1

    # Run evaluation for each model
    all_results: dict[str, list[MetricResult]] = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        try:
            if args.image_dir and len(models) == 1:
                image_dir = args.image_dir
            else:
                image_dir = resolve_image_dir(
                    image_dir=None,
                    resolution=args.resolution,
                    model=model_name,
                    samples_root=args.samples_root,
                )
            print(f"  Image directory: {image_dir}")

            n_images = len(discover_images(image_dir))
            print(f"  Found {n_images} images")
            if n_images == 0:
                print(f"  Skipping {model_name}: no images.", file=sys.stderr)
                continue

            results = run_single_evaluation(
                image_dir=image_dir,
                metric_names=metric_names,
                prompts_path=args.prompts_path,
                reference_dir=args.reference_dir,
                device=args.device,
                batch_size=args.batch_size,
                max_images=args.max_images,
            )

            key = f"{args.resolution}/{model_name}" if args.resolution else model_name
            all_results[key] = results

        except Exception as e:
            print(f"  ERROR for {model_name}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    # Output
    print_results_table(all_results)

    if args.output:
        save_results(all_results, args.output)

    if args.save_per_image:
        save_per_image_results(all_results, args.save_per_image)

    return 0


if __name__ == "__main__":
    sys.exit(main())
