#!/usr/bin/env python3
"""
Generate manifest files for each (model, resolution) task.
Each manifest lists prompt indices (0-based) that should be run.
All jobs read the same manifests; "done" is inferred from output files.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_resolutions import RESOLUTION_CONFIG
from model_handler import MODEL_REGISTRY, list_models


def count_prompts(prompts_path: Path) -> int:
    n = 0
    with open(prompts_path) as f:
        for line in f:
            if not line.strip():
                continue
            json.loads(line)
            n += 1
    return n


def main() -> int:
    manifests_dir = Path(__file__).resolve().parent / "manifests"
    manifests_dir.mkdir(exist_ok=True)

    models = [m for m in list_models(implemented_only=True) if m in MODEL_REGISTRY]
    if not models:
        print("No implemented models found")
        return 1

    for res_name, (h, w, prompts_path) in RESOLUTION_CONFIG.items():
        if not prompts_path.exists():
            print(f"Warning: {prompts_path} not found, skipping {res_name}")
            continue
        n = count_prompts(prompts_path)
        for model in models:
            manifest_path = manifests_dir / f"{model}_{res_name}.txt"
            with open(manifest_path, "w") as f:
                for i in range(n):
                    f.write(f"{i}\n")
            print(f"Wrote {manifest_path} ({n} prompts)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
