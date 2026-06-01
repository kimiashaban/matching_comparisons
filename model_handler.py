"""
Model registry and command construction for matching_comparisons.

This repo intentionally exposes only the requested methods:
ScaleDiff, I-Max, HiFlow, DiffuseHigh, DyPE, DyPE-Qwen, FreCaS, FreeScale.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXTERNAL_REPOS = ROOT / "external_repos"


@dataclass
class ModelConfig:
    name: str
    backend: str
    repo_path: Path | str
    script: str
    default_height: int
    default_width: int
    steps: int
    max_prompt_tokens: int
    method: str | None = None
    prompt_suffix: str = ""
    implemented: bool = True
    extra_args: dict = field(default_factory=dict)

    def resolved_repo_path(self) -> Path:
        repo = Path(self.repo_path)
        return repo if repo.is_absolute() else EXTERNAL_REPOS / repo


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "ScaleDiff": ModelConfig(
        name="ScaleDiff",
        backend="flux",
        repo_path="ScaleDiff/FLUX",
        script="run_eval.py",
        default_height=4096,
        default_width=4096,
        steps=28,
        max_prompt_tokens=256,
    ),
    "I-Max": ModelConfig(
        name="I-Max",
        backend="flux",
        repo_path="I-Max",
        script="run_eval.py",
        default_height=4096,
        default_width=4096,
        steps=28,
        max_prompt_tokens=512,
    ),
    "HiFlow": ModelConfig(
        name="HiFlow",
        backend="flux",
        repo_path="HiFlow",
        script="run_eval.py",
        default_height=4096,
        default_width=4096,
        steps=28,
        max_prompt_tokens=512,
    ),
    "DiffuseHigh": ModelConfig(
        name="DiffuseHigh",
        backend="sdxl",
        repo_path="DiffuseHigh",
        script="run_eval.py",
        default_height=4096,
        default_width=4096,
        steps=40,
        max_prompt_tokens=77,
    ),
    "DyPE": ModelConfig(
        name="DyPE",
        backend="flux",
        repo_path="DyPE",
        script="run_dype.py",
        method="yarn",
        default_height=4096,
        default_width=4096,
        steps=28,
        max_prompt_tokens=512,
        extra_args={"guidance_scale": 4.5},
    ),
    "DyPE-Qwen": ModelConfig(
        name="DyPE-Qwen",
        backend="qwen",
        repo_path="DyPE",
        script="run_dype_qwen.py",
        method="dype",
        default_height=3072,
        default_width=3072,
        steps=40,
        max_prompt_tokens=512,
    ),
    "FreCaS": ModelConfig(
        name="FreCaS",
        backend="sdxl",
        repo_path="FreCaS",
        script="main.py",
        default_height=4096,
        default_width=4096,
        steps=50,
        max_prompt_tokens=77,
    ),
    "FreeScale": ModelConfig(
        name="FreeScale",
        backend="sdxl",
        repo_path="FreeScale",
        script="run_eval.py",
        default_height=4096,
        default_width=4096,
        steps=50,
        max_prompt_tokens=77,
    ),
}


def get_full_prompt(model_name: str, base_prompt: str) -> str:
    cfg = MODEL_REGISTRY.get(model_name)
    if not cfg or not cfg.prompt_suffix:
        return base_prompt
    if base_prompt.endswith(cfg.prompt_suffix.strip()):
        return base_prompt
    return base_prompt.rstrip() + cfg.prompt_suffix


def build_run_command(
    model_name: str,
    prompt: str,
    *,
    output_dir: Path | str | None = None,
    height: int | None = None,
    width: int | None = None,
    steps: int | None = None,
    seed: int = 42,
    no_dype: bool = False,
    multi_gpu: bool = False,
    input_path: str | None = None,
) -> tuple[list[str], Path]:
    del input_path
    cfg = MODEL_REGISTRY.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown model: {model_name}. Known: {list(MODEL_REGISTRY)}")
    if not cfg.implemented:
        raise NotImplementedError(f"Model is registered but disabled: {model_name}")

    cwd = cfg.resolved_repo_path()
    if not cwd.exists():
        raise FileNotFoundError(f"External repo not found: {cwd}")

    full_prompt = get_full_prompt(model_name, prompt)
    h = height if height is not None else cfg.default_height
    w = width if width is not None else cfg.default_width
    s = steps if steps is not None else cfg.steps
    out_dir = Path(output_dir) if output_dir else ROOT / "outputs" / model_name

    if model_name == "DyPE":
        cmd = [
            "python", cfg.script,
            "--prompt", full_prompt,
            "--height", str(h),
            "--width", str(w),
            "--steps", str(s),
            "--seed", str(seed),
            "--method", cfg.method or "yarn",
            "--output_dir", str(out_dir),
        ]
        if no_dype:
            cmd.append("--no_dype")
    elif model_name == "DyPE-Qwen":
        cmd = [
            "python", cfg.script,
            "--prompt", full_prompt,
            "--height", str(h),
            "--width", str(w),
            "--steps", str(s),
            "--seed", str(seed),
            "--method", cfg.method or "dype",
            "--output_dir", str(out_dir),
        ]
    elif model_name == "FreCaS":
        if h == 1024 and w == 1024:
            tsize = "[[512,512],[1024,1024]]"
            msp_endtimes = ["100", "0"]
            msp_steps = ["40", "10"]
            msp_gamma = "3.0"
            facfg_weight = ["45.0", "7.5"]
            vae_tiling = False
        elif h == 2048 and w == 2048:
            tsize = "[[1024,1024],[2048,2048]]"
            msp_endtimes = ["200", "0"]
            msp_steps = ["40", "10"]
            msp_gamma = "1.5"
            facfg_weight = ["35.0", "7.5"]
            vae_tiling = False
        elif h == 4096 and w == 4096:
            tsize = "[[1024,1024],[2048,2048],[4096,4096]]"
            msp_endtimes = ["400", "200", "0"]
            msp_steps = ["30", "5", "15"]
            msp_gamma = "2.0"
            facfg_weight = ["35.0", "7.5"]
            vae_tiling = True
        else:
            tsize = f"[[1024,1024],[{h},{w}]]"
            msp_endtimes = ["200", "0"]
            msp_steps = ["40", "10"]
            msp_gamma = "1.5"
            facfg_weight = ["35.0", "7.5"]
            vae_tiling = h >= 4096 or w >= 4096

        cmd = [
            "python", cfg.script,
            "--name", "sdxl",
            "--prompts", full_prompt,
            "--output_dir", str(out_dir),
            "--tsize", tsize,
            "--msp_endtimes", *msp_endtimes,
            "--msp_steps", *msp_steps,
            "--msp_gamma", msp_gamma,
            "--gs", "7.5",
            "--images-per-prompt", "1",
            "--facfg_weight", *facfg_weight,
            "--camap_weight", "0.6",
        ]
        if vae_tiling:
            cmd.append("--vae_tiling")
    else:
        cmd = [
            "python", cfg.script,
            "--prompt", full_prompt,
            "--height", str(h),
            "--width", str(w),
            "--seed", str(seed),
            "--output_dir", str(out_dir),
        ]

    if multi_gpu:
        cmd.append("--multi_gpu")
    return cmd, cwd


def run_model(
    model_name: str,
    prompt: str,
    *,
    output_dir: Path | str | None = None,
    height: int | None = None,
    width: int | None = None,
    steps: int | None = None,
    seed: int = 42,
    no_dype: bool = False,
    multi_gpu: bool = False,
    dry_run: bool = False,
    env: dict | None = None,
    prompt_index: int | None = None,
) -> subprocess.CompletedProcess | None:
    del prompt_index
    cmd, cwd = build_run_command(
        model_name,
        prompt,
        output_dir=output_dir,
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        no_dype=no_dype,
        multi_gpu=multi_gpu,
    )
    print(f"[{model_name}] cwd={cwd}")
    print(f"[{model_name}] {' '.join(cmd)}")
    if dry_run:
        return None
    return subprocess.run(cmd, cwd=str(cwd), env={**os.environ, **(env or {})})


def list_models(*, implemented_only: bool = False, **_: object) -> list[str]:
    names = list(MODEL_REGISTRY)
    if implemented_only:
        names = [name for name in names if MODEL_REGISTRY[name].implemented]
    return names
