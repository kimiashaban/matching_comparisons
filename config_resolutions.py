"""
Resolution config for Aesthetic-4K evaluation.
Maps resolution strings (e.g. 4096x4096) to height, width, prompts file.
"""

from pathlib import Path

HIRES_EVAL_ROOT = Path(__file__).resolve().parent
DATA_ROOT = HIRES_EVAL_ROOT / "data"
A4K_METADATA = DATA_ROOT / "aesthetic4k_metadata.jsonl"
ZERO_SHOT_BENCHMARK = DATA_ROOT / "zero_shot_high_res_image_gen_benchmark.jsonl"

# Resolution name -> (height, width, prompts_file_path)
# All Aesthetic-4K resolutions use the 4096 prompt list (195 prompts).
# zero_shot_4096x4096 uses the zero-shot benchmark (200 prompts).
RESOLUTION_CONFIG = {
    "2048x4096": (2048, 4096, A4K_METADATA),
    "3072x3072": (3072, 3072, A4K_METADATA),
    "4096x4096": (4096, 4096, A4K_METADATA),
    "4096x2048": (4096, 2048, A4K_METADATA),
    "zero_shot_4096x4096": (4096, 4096, ZERO_SHOT_BENCHMARK),
}

# Suggested batch size by resolution (lower for larger images to save VRAM)
DEFAULT_BATCH_SIZE_BY_RES = {
    "2048x4096": 4,
    "3072x3072": 2,
    "4096x4096": 2,
    "4096x2048": 2,
}
