"""
CLI wrapper for ScaleDiff (FLUX.1-dev) evaluation.
Mirrors the interface of other run_eval.py wrappers in this benchmark.
"""
import argparse
import os
from pathlib import Path

import torch
from pipeline_scalediff_flux import FluxPipeline
from transformer_scalediff_flux import FluxTransformer2DModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--height", type=int, default=4096)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--steps", type=int, default=28)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ScaleDiff takes a base resolution and upsamples by 2x per upsample_stage.
    # Output = base * 2^upsample_stage. We pass target size from run_task.py,
    # so divide by 4 (2^2) to get the correct base for upsample_stage=2.
    upsample_stage = 2
    base_height = args.height // 4
    base_width = args.width // 4

    device = "cuda"
    model_path = "black-forest-labs/FLUX.1-dev"

    transformer = FluxTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = FluxPipeline.from_pretrained(
        model_path, transformer=transformer, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.vae.enable_tiling()

    generator = torch.Generator(device=device).manual_seed(args.seed)

    image = pipe(
        prompt=args.prompt,
        height=base_height,
        width=base_width,
        guidance_scale=3.5,
        num_inference_steps=args.steps,
        max_sequence_length=256,
        generator=generator,
        restart_ratio=0.6,
        scale_factor=0.25,
        upsample_stage=upsample_stage,
        query_random_jitter=True,
        t5_to_cpu=True,
    )[-1]

    out_path = Path(args.output_dir) / f"seed_{args.seed}_{args.height}x{args.width}.png"
    image.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
