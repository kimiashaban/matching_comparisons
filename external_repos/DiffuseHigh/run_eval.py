"""
CLI wrapper for DiffuseHigh evaluation.
"""
import argparse
import os

import torch
from pipeline_diffusehigh_sdxl import DiffuseHighSDXLPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str,
        default="blurry, ugly, duplicate, poorly drawn, deformed, mosaic")
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generator = torch.Generator("cuda").manual_seed(args.seed)

    pipeline = DiffuseHighSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    ).to("cuda")

    target_h = [min(1536, args.height), args.height] if args.height > 1536 else [args.height]
    target_w = [min(1536, args.width), args.width] if args.width > 1536 else [args.width]

    image = pipeline(
        args.prompt,
        negative_prompt=args.negative_prompt,
        target_height=target_h,
        target_width=target_w,
        enable_dwt=True,
        dwt_steps=5,
        enable_sharpening=True,
        sharpness_factor=1.0,
        generator=generator,
    ).images[0]

    out_path = os.path.join(args.output_dir, f"seed_{args.seed}_res_{args.height}x{args.width}.png")
    image.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
