#!/usr/bin/env python3
"""
Run a single (model, resolution) task with manifest-based prompt tracking.
Reads manifest to know all prompts; infers "done" from output dir; runs only remaining.
Supports parallel jobs via --job-index and --total-jobs.

Usage:
  python run_task.py --model ScaleDiff --resolution 4096x4096 --seed 12345
  python run_task.py --model DyPE --resolution 4096x4096 --job-index 0 --total-jobs 2
  python run_task.py --model FreCaS --resolution 2048x4096 --dry-run
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_resolutions import (
    DEFAULT_BATCH_SIZE_BY_RES,
    RESOLUTION_CONFIG,
)
from model_handler import (
    MODEL_REGISTRY,
    build_run_command,
    run_model,
)


def load_prompts(prompts_path: Path) -> list[str]:
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompts.append(row["text"])
    return prompts


def load_manifest(manifest_path: Path) -> list[int]:
    indices = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indices.append(int(line))
    return sorted(indices)


def get_done_indices(output_dir: Path) -> set[int]:
    """Scan output dir for NNNNN_*.png (or jpg etc) and extract prompt indices."""
    done = set()
    if not output_dir.exists():
        return done
    pat = re.compile(r"^(\d{5})_[^/]+\.(png|jpg|jpeg|webp)$", re.I)
    for p in output_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            done.add(int(m.group(1)))
    return done


def slugify_filename(text: str, max_len: int = 120) -> str:
    s = text.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    s = s.strip("._-")
    return (s or "prompt")[:max_len]


def collect_image_files(dir_path: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a (model, resolution) task with manifest-based resume")
    parser.add_argument("--model", "-m", required=True, help="Model name")
    parser.add_argument("--resolution", "-r", required=True, help="e.g. 4096x4096, 2048x2048")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=None, help="Default: 2 for 4096, 4 for 2048")
    parser.add_argument("--job-index", type=int, default=0, help="For parallel: this job's index (0-based)")
    parser.add_argument("--total-jobs", type=int, default=1, help="For parallel: total number of jobs")
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--output-base", type=Path, default=None)
    parser.add_argument("--multi-gpu", action="store_true", help="Pass --multi_gpu to model scripts")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    manifest_dir = args.manifest_dir or (root / "manifests")
    output_base = args.output_base or (root / "outputs_by_res")

    if args.model not in MODEL_REGISTRY or not MODEL_REGISTRY[args.model].implemented:
        print(f"Unknown or unimplemented model: {args.model}", file=sys.stderr)
        return 1

    if args.resolution not in RESOLUTION_CONFIG:
        print(f"Unknown resolution: {args.resolution}. Known: {list(RESOLUTION_CONFIG)}", file=sys.stderr)
        return 1

    h, w, prompts_path = RESOLUTION_CONFIG[args.resolution]
    if not prompts_path.exists():
        print(f"Prompts file not found: {prompts_path}", file=sys.stderr)
        return 1

    manifest_path = manifest_dir / f"{args.model}_{args.resolution}.txt"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}. Run: python generate_manifests.py", file=sys.stderr)
        return 1

    all_indices = load_manifest(manifest_path)
    prompts = load_prompts(prompts_path)

    # Zero-shot benchmark outputs go under outputs_by_res/zero_shot/<model>/
    # Standard resolutions go under outputs_by_res/<resolution>/<model>/
    if args.resolution.startswith("zero_shot_"):
        model_dir = output_base / "zero_shot" / args.model
    else:
        model_dir = output_base / args.resolution / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    done = get_done_indices(model_dir)

    remaining = [i for i in all_indices if i not in done]
    if not remaining:
        print(f"[{args.model} {args.resolution}] All {len(all_indices)} prompts done.")
        return 0

    # Slice remaining by JOB_INDEX / TOTAL_JOBS for parallel runs
    total_jobs = max(1, args.total_jobs)
    job_idx = max(0, min(args.job_index, total_jobs - 1))
    chunk = (len(remaining) + total_jobs - 1) // total_jobs
    start = job_idx * chunk
    end = min(start + chunk, len(remaining))
    my_remaining = remaining[start:end]

    if not my_remaining:
        print(f"[{args.model} {args.resolution}] {len(done)} done, {len(remaining)} remaining. Job {job_idx}/{total_jobs} has no prompts in its slice.")
        return 0

    print(f"[{args.model} {args.resolution}] {len(done)} done, {len(remaining)} remaining. Job {job_idx}/{total_jobs} running {len(my_remaining)} prompts: {my_remaining[0]}..{my_remaining[-1]}", flush=True)

    slurm_tmp = os.environ.get("SLURM_TMPDIR") or "/tmp"
    job_tag = os.environ.get("SLURM_JOB_ID") or "interactive"
    tmp_root = Path(slurm_tmp) / "hires_eval_tmp" / job_tag / args.model

    for idx in my_remaining:
        prompt = prompts[idx]
        tmp_dir = tmp_root / f"prompt_{idx:05d}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {args.model} {args.resolution} | prompt {idx + 1}/{len(prompts)} (index {idx}) ---", flush=True)
        try:
            before = set(collect_image_files(tmp_dir))
            result = run_model(
                args.model,
                prompt,
                output_dir=str(tmp_dir),
                height=h,
                width=w,
                steps=None,
                seed=args.seed,
                no_dype=False,
                multi_gpu=args.multi_gpu,
                dry_run=args.dry_run,
                prompt_index=idx,
            )
            if args.dry_run:
                continue
            if result and result.returncode != 0:
                print(f"Failed prompt {idx}", file=sys.stderr)
                continue

            after = collect_image_files(tmp_dir)
            new_imgs = [p for p in after if p not in before]
            if not new_imgs:
                print(f"Warning: no images for prompt {idx}", file=sys.stderr)
                continue

            slug = slugify_filename(prompt)
            for oi, src in enumerate(new_imgs):
                dst_name = f"{idx:05d}_{slug}"
                if len(new_imgs) > 1:
                    dst_name += f"_{oi:02d}"
                dst = model_dir / f"{dst_name}{src.suffix.lower()}"
                shutil.move(str(src), str(dst))
                print(f"Saved to: {dst}")

            leftovers = list(tmp_dir.iterdir())
            if not leftovers:
                tmp_dir.rmdir()
        except Exception as e:
            print(f"Error prompt {idx}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
