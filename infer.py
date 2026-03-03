#!/usr/bin/env python3
"""
Single-image background replacement using Flux1-Fill-dev (mflux / Apple Silicon).

Usage:
    python infer.py --image JD_1K/bg1k_imgs/0.png \
                    --mask  JD_1K/bg1k_masks/0_mask.png \
                    --prompt "modern minimalist kitchen, soft natural light" \
                    --output output.png

Mask convention (BG-1K RGBA masks):
    Alpha > 128  → foreground / product  → PRESERVE  (black in fill mask)
    Alpha ≤ 128  → background            → FILL       (white in fill mask)
"""

import argparse
import os
import tempfile

from PIL import Image
import numpy as np


# ── mask conversion ─────────────────────────────────────────────────────────

def make_fill_mask(mask_path: str, tmp_dir: str) -> str:
    """Convert a BG-1K RGBA mask to a binary fill mask (white=fill, black=keep)."""
    mask = Image.open(mask_path).convert("RGBA")
    alpha = np.array(mask)[:, :, 3]                    # foreground alpha
    fill = np.where(alpha > 128, 0, 255).astype(np.uint8)  # invert: bg → white
    fill_img = Image.fromarray(fill, mode="L")
    out_path = os.path.join(tmp_dir, "fill_mask.png")
    fill_img.save(out_path)
    return out_path


def resize_to_multiple(image_path: str, tmp_dir: str, multiple: int = 64) -> tuple[str, tuple[int, int]]:
    """Ensure image dimensions are multiples of `multiple` (required by Flux)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
        out_path = os.path.join(tmp_dir, "resized_image.png")
        img.save(out_path)
        return out_path, (new_w, new_h)
    return image_path, (w, h)


# ── inference ────────────────────────────────────────────────────────────────

def run_inference(
    image_path: str,
    mask_path: str,
    prompt: str,
    output_path: str,
    seed: int = 42,
    steps: int = 28,
    guidance: float = 30.0,
    quantize: int = 8,
):
    from mflux.models.flux.variants.fill.flux_fill import Flux1Fill

    with tempfile.TemporaryDirectory() as tmp:
        fill_mask_path = make_fill_mask(mask_path, tmp)
        resized_image_path, (width, height) = resize_to_multiple(image_path, tmp)

        print(f"Loading Flux1-Fill-dev (quantize={quantize})...")
        flux = Flux1Fill(quantize=quantize)

        print(f"Generating background — seed={seed}, steps={steps}, guidance={guidance}")
        print(f"  prompt : {prompt}")
        print(f"  size   : {width}x{height}")

        image = flux.generate_image(
            seed=seed,
            prompt=prompt,
            width=width,
            height=height,
            guidance=guidance,
            image_path=resized_image_path,
            num_inference_steps=steps,
            masked_image_path=fill_mask_path,
        )

        image.save(path=output_path)
        print(f"Saved → {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Flux1-Fill background replacement (single image)")
    p.add_argument("--image",    required=True,  help="Path to product image (RGB PNG/JPG)")
    p.add_argument("--mask",     required=True,  help="Path to BG-1K RGBA mask")
    p.add_argument("--prompt",   required=True,  help="Text description of the desired background")
    p.add_argument("--output",   default="output.png", help="Output file path (default: output.png)")
    p.add_argument("--seed",     type=int,   default=42,   help="RNG seed")
    p.add_argument("--steps",    type=int,   default=28,   help="Inference steps (default: 28)")
    p.add_argument("--guidance", type=float, default=30.0, help="Guidance scale (default: 30)")
    p.add_argument("--quantize", type=int,   default=8,    choices=[4, 6, 8],
                   help="Weight quantization bits (default: 8, use 4 to save VRAM)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        image_path=args.image,
        mask_path=args.mask,
        prompt=args.prompt,
        output_path=args.output,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        quantize=args.quantize,
    )
