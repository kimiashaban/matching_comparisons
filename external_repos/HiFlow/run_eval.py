"""
CLI wrapper for HiFlow evaluation.
"""
import argparse
import os

import torch
from flux_pipeline_hiflow import FluxPipeline
from transformer_flux import FluxTransformer2DModel
from utils import set_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seeds(args.seed)

    device = "cuda"
    model_path = "black-forest-labs/FLUX.1-dev"

    transformer = FluxTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.float16
    )
    pipe = FluxPipeline.from_pretrained(
        model_path, transformer=None, torch_dtype=torch.float16
    )
    pipe.transformer = transformer
    # pipe.scheduler.use_dynamic_shifting = False
    pipe.to(device)

    # flux_pipeline_hiflow zips target_heights with target_widths; lists MUST be the
    # same length. Old logic used independent [2048] vs [2048, w] lists for rectangles,
    # which truncated to one (2048, 2048) upscale and wrong outputs.
    h, w = args.height, args.width
    if max(h, w) <= 1024:
        target_h, target_w = [1024], [1024]
    else:
        # Match square case: (2048, 2048) then (h, w); works for 2048x4096, 4096x2048, etc.
        target_h, target_w = [1024, h], [1024, w]
    n_up = len(target_h)
    steps_hr = [5] * n_up

    images = pipe(
        prompt=args.prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=28,
        max_sequence_length=512,
        ntk_factor=[10.0] * n_up,
        proportional_attention=True,
        text_duplication=False,
        swin_pachify=True,
        target_heights=target_h,
        target_widths=target_w,
        num_inference_steps_highres=steps_hr,
        filter_ratio=[0.2] * n_up,
        guidance_scale_highres=[4.5] * n_up,
        structure_guidance="fft",
        alphas=[1.0] * n_up,
        betas=[0.5] * n_up,
        upsampling_choice="latent",
        flow_choice="v_theta",
        generator=torch.Generator("cuda").manual_seed(args.seed),
    )[0]

    out_path = os.path.join(args.output_dir, f"seed_{args.seed}_res_{args.height}x{args.width}.png")
    images[0].save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
