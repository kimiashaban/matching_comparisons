"""
Official implementation for ultra-high resolution image generation with Qwen
DyPE: Dynamic Position Extrapolation for Ultra High Resolution Diffusion
"""

import os
import sys
import types
import argparse
import torch
from diffusers import DiffusionPipeline

sys.path.append(os.getcwd())
try:
    from qwen.transformer_qwenimage import QwenImageTransformer2DModel
except ImportError:
    print("Could not import QwenImageTransformer from qwen.transformer_qwenimage")
    print("Ensure the file exists and the class name is correct.")
    sys.exit(1)

GPU_MEMORY_THRESHOLD_GB = 60


def get_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def _move_to_device(v, device):
    """Recursively move tensors to device, handling tuples, lists, dicts, and ModelOutput."""
    if isinstance(v, torch.Tensor):
        return v.to(device)
    elif isinstance(v, tuple):
        return tuple(_move_to_device(x, device) for x in v)
    elif isinstance(v, list):
        return [_move_to_device(x, device) for x in v]
    elif isinstance(v, dict):
        moved = {k: _move_to_device(val, device) for k, val in v.items()}
        if type(v) is not dict:
            try:
                return type(v)(**moved)
            except Exception:
                pass
        return moved
    return v


def main():
    parser = argparse.ArgumentParser(
        description='DyPE: Generate ultra-high resolution images with Qwen'
    )

    # 1. Specific Prompt Default
    parser.add_argument(
        '--prompt',
        type=str,
        default="A woman kneels on the forest floor, smiling as she offers grapes to a large brown bear beside her, surrounded by tall birch trees.",
        help='Text prompt for image generation'
    )

    # 2. Resolution Default (3072x3072)
    parser.add_argument('--height', type=int, default=3072, help='Image height in pixels')
    parser.add_argument('--width', type=int, default=3072, help='Image width in pixels')

    # 3. Steps Default (40)
    parser.add_argument('--steps', type=int, default=40, help='Number of inference steps')

    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    # 4. Method logic (dype vs base)
    parser.add_argument(
        '--method',
        type=str,
        choices=['dype', 'base'],
        default='dype',
        help='Position encoding method (dype or base)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Output directory for generated images'
    )
    parser.add_argument(
        '--multi_gpu',
        action='store_true',
        help='Distribute transformer blocks across all available GPUs',
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = "Qwen/Qwen-Image"

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    num_gpus = torch.cuda.device_count()
    enable_multi_gpu = args.multi_gpu and num_gpus >= 2

    if enable_multi_gpu:
        print(f"[info] Multi-GPU enabled: distributing Qwen transformer across {num_gpus} GPUs")

    use_dype = (args.method == 'dype')

    print(f"Loading {model_name}...")

    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        dype=use_dype,
        low_cpu_mem_usage=True,
    )

    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )

    pipe.vae.enable_tiling()

    if enable_multi_gpu:
        transformer = pipe.transformer

        for name, child in transformer.named_children():
            if name != "transformer_blocks":
                child.to("cuda:0")
                print(f"  {name} -> cuda:0")

        blocks_per_gpu = len(transformer.transformer_blocks) // num_gpus
        for i, block in enumerate(transformer.transformer_blocks):
            device = f"cuda:{min(i // blocks_per_gpu, num_gpus - 1)}"
            block.to(device)
            block._original_forward = block.forward.__func__
            block._target_device = device

            def forward_wrapper(self, *args, **kwargs):
                target_dev = self._target_device
                args = tuple(_move_to_device(a, target_dev) for a in args)
                kwargs = {k: _move_to_device(v, target_dev) for k, v in kwargs.items()}
                return _move_to_device(self._original_forward(self, *args, **kwargs), "cuda:0")

            block.forward = types.MethodType(forward_wrapper, block)
            print(f"  Transformer block {i} -> {device}")

        transformer._original_transformer_forward = transformer.forward.__func__

        def transformer_forward_wrapper(self, *args, **kwargs):
            args = tuple(_move_to_device(a, "cuda:0") for a in args)
            kwargs = {k: _move_to_device(v, "cuda:0") for k, v in kwargs.items()}
            result = self._original_transformer_forward(self, *args, **kwargs)
            if hasattr(result, 'sample'):
                result.sample = result.sample.to("cuda:0")
            return result

        transformer.forward = types.MethodType(transformer_forward_wrapper, transformer)
        print("[info] Transformer forward wrapped for multi-GPU device management")

        pipe.text_encoder.to("cpu")
        pipe.vae.to("cuda:0")
        print(f"[info] VAE on cuda:0, text encoder on CPU, transformer across {num_gpus} GPUs")

        # Hard-wire _execution_device to cuda:0. Because text_encoder is the first
        # registered component, diffusers' device property returns cpu, causing latent
        # tensors to be created on CPU — a fatal mismatch with the cuda:0 generator.
        _orig_cls = type(pipe)
        pipe.__class__ = type(
            _orig_cls.__name__,
            (_orig_cls,),
            {"_execution_device": property(lambda self: torch.device("cuda:0"))},
        )
        print("[info] Patched _execution_device -> cuda:0")

        # Text encoder device bridge: the pipeline sends tokenized inputs to
        # _execution_device (cuda:0), but text_encoder weights live on CPU.
        _real_te_forward = pipe.text_encoder.forward

        def _te_forward_cpu_bridge(*args, **kwargs):
            args = tuple(_move_to_device(a, "cpu") for a in args)
            kwargs = {k: _move_to_device(v, "cpu") for k, v in kwargs.items()}
            return _move_to_device(_real_te_forward(*args, **kwargs), "cuda:0")

        pipe.text_encoder.forward = _te_forward_cpu_bridge
        print("[info] Wrapped text_encoder forward with CPU<->cuda:0 bridge")
    else:
        if get_gpu_memory_gb() < GPU_MEMORY_THRESHOLD_GB:
            print(f"[info] CPU offload enabled (GPU memory: {get_gpu_memory_gb():.1f}GB < {GPU_MEMORY_THRESHOLD_GB}GB)")
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

    positive_magic = ", Ultra HD, 4K, cinematic composition."
    full_prompt = args.prompt + positive_magic
    negative_prompt = ""

    if enable_multi_gpu:
        gen_device = "cuda:0"
    elif torch.cuda.is_available():
        gen_device = "cuda"
    else:
        gen_device = "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(args.seed)

    print(f"Generating {args.height}x{args.width} image with {args.steps} steps using Qwen (Method: {args.method})...")

    # Generate image using Qwen specific parameters
    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        true_cfg_scale=4.0,
        generator=generator
    ).images[0]

    multigpu_suffix = "_multigpu" if enable_multi_gpu else ""
    filename = os.path.join(args.output_dir, f"seed_{args.seed}_method_{args.method}{multigpu_suffix}_res_{args.height}x{args.width}.png")
    image.save(filename)
    print(f"✓ Image saved to: {filename}")


if __name__ == "__main__":
    main()