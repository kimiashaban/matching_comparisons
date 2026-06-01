"""
CLI wrapper for I-Max evaluation.
"""
import argparse
import os

import torch
from pipeline_flux_imax import FluxPipeline
from transformer_flux import FluxTransformer2DModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    bfl_repo = "black-forest-labs/FLUX.1-dev"
    transformer = FluxTransformer2DModel.from_pretrained(
        bfl_repo, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = FluxPipeline.from_pretrained(
        bfl_repo, transformer=None, torch_dtype=torch.bfloat16
    )
    pipe.transformer = transformer
    pipe.scheduler.config.use_dynamic_shifting = False
    pipe.to("cuda")

    images = pipe(
        prompt=args.prompt,
        num_inference_steps1=20,
        num_inference_steps2=5,
        guidance_scale1=3.5,
        guidance_scale2=4.5,
        height=args.height,
        width=args.width,
        ntk_factor=10,
        return_dict=False,
        time_shift_1=3,
        time_shift_2=10,
        dwt_level=1,
        proportional_attention=True,
        text_duplication=True,
        swin_pachify=True,
        guidance_schedule="cosine_decay",
    )
    image = images[0]
    out_path = os.path.join(args.output_dir, f"seed_{args.seed}_res_{args.height}x{args.width}.jpeg")
    image.save(out_path, "JPEG")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
