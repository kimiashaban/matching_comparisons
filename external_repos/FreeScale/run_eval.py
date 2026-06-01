"""
CLI wrapper for FreeScale evaluation.
"""
import argparse
import os

import torch
from pipeline_freescale import StableDiffusionXLPipeline
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

    resolutions_list = [[1024, 1024], [args.height, args.width]]
    cosine_scale = 2.0 if args.height <= 2048 else 1.0

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    ).to("cuda")
    register_free_upblock2d(pipe, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
    register_free_crossattn_upblock2d(pipe, b1=1.1, b2=1.2, s1=0.6, s2=0.4)

    generator = torch.Generator("cuda").manual_seed(args.seed)
    results = pipe(
        args.prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        num_inference_steps=50,
        guidance_scale=7.5,
        resolutions_list=resolutions_list,
        fast_mode=False,
        cosine_scale=cosine_scale,
    )
    image = results[-1].images[0]
    out_path = os.path.join(args.output_dir, f"seed_{args.seed}_res_{args.height}x{args.width}.png")
    image.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
