#!/usr/bin/env python3
"""
Run script for aesthetic-4K evaluation.

Usage:
  python run_aesthetic4k.py --model DyPE --prompt "A mountain landscape"
  python run_aesthetic4k.py --model ScaleDiff --prompt "A dog in a garden" --height 4096 --width 4096
  python run_aesthetic4k.py --model HiFlow --prompt "Sunset over the ocean"
  python run_aesthetic4k.py --model FreCaS --prompt "Portrait of a woman"

  # Batch: run all implemented models on prompts from Aesthetic-4K metadata
  python run_aesthetic4k.py --all --prompts-file /path/to/Aesthetic-4K/eval/size_4096/metadata.jsonl --limit 2 --height 2048 --width 2048

  python run_aesthetic4k.py --list                    # list available models
  python run_aesthetic4k.py --list --implemented     # list implemented only
  python run_aesthetic4k.py --dry-run --model DyPE --prompt "test"  # print command only
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# Ensure hires_eval root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_handler import (
    MODEL_REGISTRY,
    build_run_command,
    list_models,
    run_model,
)


def load_prompts_from_metadata(
    path: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[str]:
    """Load prompts from Aesthetic-4K metadata.jsonl (expects 'text' field per line).
    Use offset to skip the first N prompts; limit caps how many to return after offset.
    """
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompts.append(row["text"])
    if offset > 0:
        prompts = prompts[offset:]
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def slugify_filename(text: str, max_len: int = 120) -> str:
    s = text.strip().lower()
    s = re.sub(r"\s+", "_", s)
    # Keep it filesystem-friendly
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    s = s.strip("._-")
    if not s:
        s = "prompt"
    return s[:max_len]


def collect_image_files(dir_path: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run aesthetic-4K models for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model name (e.g. ScaleDiff, I-Max, HiFlow, DiffuseHigh, DyPE, DyPE-Qwen, FreCaS, FreeScale)",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Image height (default: model-specific)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Image width (default: model-specific)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of inference steps (default: model-specific)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/<model>)",
    )
    parser.add_argument(
        "--no-dype",
        action="store_true",
        help="Disable DyPE (only for DyPE model)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "--implemented",
        action="store_true",
        help="With --list: show only implemented models",
    )
    parser.add_argument(
        "--pe-only",
        action="store_true",
        help="With --list: show only PE-based models",
    )
    parser.add_argument(
        "--non-pe-only",
        action="store_true",
        help="With --list: show only non-PE models",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all implemented models (use with --prompts-file)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Path to metadata.jsonl (Aesthetic-4K format) - reads 'text' field",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of prompts to use (default: all, or 2 for quick test)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N prompts (for parallel job splitting)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=[],
        help="Models to skip (e.g. DyPE-Qwen FreeScale FreCaS)",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        default=None,
        help="Run ONLY these models (e.g. ScaleDiff DiffuseHigh I-Max HiFlow)",
    )

    args = parser.parse_args()

    if args.list:
        models = list_models(
            pe_only=args.pe_only,
            non_pe_only=args.non_pe_only,
            implemented_only=args.implemented,
        )
        for m in models:
            cfg = MODEL_REGISTRY[m]
            status = "✓" if cfg.implemented else "○"
            print(f"  {status} {m:20} backend={cfg.backend:12} ({cfg.default_height}x{cfg.default_width})")
        return 0

    # Batch mode: --all + --prompts-file
    if args.all and args.prompts_file:
        limit = args.limit if args.limit is not None else 2
        prompts = load_prompts_from_metadata(
            args.prompts_file, limit=limit, offset=args.offset
        )
        all_impl = list_models(implemented_only=True)
        if args.only:
            models = [m for m in args.only if m in all_impl]
        else:
            models = [m for m in all_impl if m not in (args.exclude or [])]
        # In batch mode, let --output-dir override the default base output directory.
        # We write into <base>/<model>/ and name files by prompt index + prompt slug.
        base_out = Path(args.output_dir).resolve() if args.output_dir else (Path(__file__).resolve().parent / "outputs")
        failed = []
        for mi, model in enumerate(models):
            for pi, prompt in enumerate(prompts):
                model_dir = base_out / model
                model_dir.mkdir(parents=True, exist_ok=True)
                # Run each prompt in a temp folder on *local* storage (not scratch),
                # because some models can hit NFS/scratch issues like
                # "Stale file handle" when saving via PIL. Then move/rename results
                # into the scratch output directory.
                slurm_tmp = os.environ.get("SLURM_TMPDIR") or "/tmp"
                job_tag = os.environ.get("SLURM_JOB_ID") or "interactive"
                global_idx = args.offset + pi
                tmp_root = Path(slurm_tmp) / "hires_eval_tmp" / job_tag / model
                tmp_dir = tmp_root / f"prompt_{global_idx:05d}"
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                print(f"[batch] tmp_dir={tmp_dir}")

                print(f"\n--- [{mi + 1}/{len(models)}] {model} | prompt {pi + 1}/{len(prompts)} ---")
                try:
                    before = set(collect_image_files(tmp_dir))
                    result = run_model(
                        model,
                        prompt,
                        output_dir=str(tmp_dir),
                        height=args.height or 2048,
                        width=args.width or 2048,
                        steps=args.steps,
                        seed=args.seed,
                        no_dype=args.no_dype,
                        dry_run=args.dry_run,
                    )
                    if result and result.returncode != 0:
                        failed.append((model, pi))
                        continue

                    if args.dry_run:
                        continue

                    after = collect_image_files(tmp_dir)
                    new_imgs = [p for p in after if p not in before]
                    if not new_imgs:
                        # Fallback: if the model wrote elsewhere or used subfolders, keep tmp for debugging
                        print(f"Warning: no images detected in {tmp_dir}", file=sys.stderr)
                        continue

                    prompt_slug = slugify_filename(prompt)
                    for oi, src in enumerate(new_imgs):
                        # use global index so parallel jobs write distinct filenames
                        dst_name = f"{global_idx:05d}_{prompt_slug}"
                        if len(new_imgs) > 1:
                            dst_name += f"_{oi:02d}"
                        dst = model_dir / f"{dst_name}{src.suffix.lower()}"
                        shutil.move(str(src), str(dst))
                        print(f"Saved to: {dst}")

                    # Clean up temp folder (keep if anything unexpected remains)
                    leftovers = [p for p in tmp_dir.iterdir()]
                    if not leftovers:
                        tmp_dir.rmdir()
                except (NotImplementedError, ValueError, FileNotFoundError) as e:
                    print(f"Error: {e}", file=sys.stderr)
                    failed.append((model, pi))
        if failed:
            print(f"\nFailed: {failed}", file=sys.stderr)
            return 1
        return 0

    if not args.model or not args.prompt:
        parser.error("--model and --prompt are required (unless using --list or --all with --prompts-file)")

    try:
        if args.dry_run:
            cmd, cwd = build_run_command(
                args.model,
                args.prompt,
                output_dir=args.output_dir,
                height=args.height,
                width=args.width,
                steps=args.steps,
                seed=args.seed,
                no_dype=args.no_dype,
            )
            print(f"cwd: {cwd}")
            print(f"cmd: {' '.join(cmd)}")
            return 0

        result = run_model(
            args.model,
            args.prompt,
            output_dir=args.output_dir,
            height=args.height,
            width=args.width,
            steps=args.steps,
            seed=args.seed,
            no_dype=args.no_dype,
            dry_run=False,
        )
        return result.returncode if result else 1

    except NotImplementedError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
