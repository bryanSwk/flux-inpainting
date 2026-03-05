import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def find_pairs(orig_dir: Path, gen_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return [(id, orig_path, gen_path)] for every matched pair."""
    pairs = []
    for gen_path in sorted(gen_dir.glob("*.png")):
        img_id = gen_path.stem
        orig_path = orig_dir / f"{img_id}.png"
        if not orig_path.exists():
            # try .jpg fallback
            orig_path = orig_dir / f"{img_id}.jpg"
        if orig_path.exists():
            pairs.append((img_id, orig_path, gen_path))
    return pairs


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ── CLIP similarity ───────────────────────────────────────────────────────────

def clip_image_features(images: list[Image.Image], processor, model, device: str, batch_size: int = 32) -> torch.Tensor:
    """Return L2-normalised CLIP image embeddings, shape (N, D)."""
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            # Use vision_model + visual_projection for compatibility with transformers v5
            vis_out = model.vision_model(pixel_values=inputs["pixel_values"])
            feats = model.visual_projection(vis_out.pooler_output)
            feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def compute_clip_similarities(
    orig_imgs: list[Image.Image],
    gen_imgs: list[Image.Image],
    model_name: str,
    device: str,
) -> list[float]:
    print(f"\nLoading CLIP model '{model_name}' on {device}...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    print("Extracting CLIP features for original images...")
    orig_feats = clip_image_features(orig_imgs, processor, model, device)
    print("Extracting CLIP features for generated images...")
    gen_feats = clip_image_features(gen_imgs, processor, model, device)

    # Per-pair cosine similarity (embeddings already normalised → dot product)
    sims = (orig_feats * gen_feats).sum(dim=-1).tolist()
    return sims


# ── FID ───────────────────────────────────────────────────────────────────────

# FID InceptionV3 expects uint8 tensors in range [0, 255], shape (N, 3, H, W)
_to_tensor_uint8 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),                      # float32 [0, 1]
    transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
])


def compute_fid(orig_imgs: list[Image.Image], gen_imgs: list[Image.Image], device: str) -> float:
    # FID on MPS is unsupported by some Inception ops; fall back to CPU
    fid_device = device if device != "mps" else "cpu"
    print(f"\nComputing FID on {fid_device} (InceptionV3)...")

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(fid_device)

    def feed(images: list[Image.Image], real: bool, desc: str):
        batch_size = 16
        for i in tqdm(range(0, len(images), batch_size), desc=desc, leave=False):
            batch = images[i : i + batch_size]
            tensors = torch.stack([_to_tensor_uint8(img) for img in batch]).to(fid_device)
            fid.update(tensors, real=real)

    feed(orig_imgs, real=True,  desc="  FID orig")
    feed(gen_imgs,  real=False, desc="  FID gen ")
    return fid.compute().item()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated images: CLIP similarity + FID")
    parser.add_argument("--orig-dir",    default="JD_1K/bg1k_imgs", help="Original images directory")
    parser.add_argument("--gen-dir",     default="outputs",          help="Generated images directory")
    parser.add_argument("--clip-model",  default="openai/clip-vit-large-patch14",
                        help="HuggingFace CLIP model ID")
    parser.add_argument("--top-k",       type=int, default=5,
                        help="Show best and worst N pairs by CLIP similarity")
    parser.add_argument("--save-csv",    default=None, help="Save per-image results to CSV")
    parser.add_argument("--save-json",   default=None, help="Save summary to JSON")
    parser.add_argument("--no-fid",      action="store_true", help="Skip FID computation")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    orig_dir = Path(args.orig_dir)
    gen_dir  = Path(args.gen_dir)

    # ── discover pairs ──
    pairs = find_pairs(orig_dir, gen_dir)
    if not pairs:
        print(f"No matching pairs found between '{orig_dir}' and '{gen_dir}'.")
        return
    print(f"Found {len(pairs)} matched image pairs.")

    ids        = [p[0] for p in pairs]
    orig_imgs  = [load_rgb(p[1]) for p in tqdm(pairs, desc="Loading originals")]
    gen_imgs   = [load_rgb(p[2]) for p in tqdm(pairs, desc="Loading generated")]

    # ── CLIP similarity ──
    sims = compute_clip_similarities(orig_imgs, gen_imgs, args.clip_model, device)

    mean_sim = sum(sims) / len(sims)
    min_sim  = min(sims)
    max_sim  = max(sims)

    # ── FID ──
    fid_score = None
    if not args.no_fid:
        fid_score = compute_fid(orig_imgs, gen_imgs, device)

    # ── print summary ──
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Pairs evaluated  : {len(pairs)}")
    print(f"  CLIP similarity  : mean={mean_sim:.4f}  min={min_sim:.4f}  max={max_sim:.4f}")
    if fid_score is not None:
        print(f"  FID score        : {fid_score:.2f}  (lower is better)")
    print("=" * 50)

    # ── best / worst ──
    ranked = sorted(zip(ids, sims), key=lambda x: x[1])
    print(f"\nWorst {args.top_k} pairs (lowest CLIP similarity):")
    for img_id, score in ranked[: args.top_k]:
        print(f"  {img_id:<10}  {score:.4f}")

    print(f"\nBest {args.top_k} pairs (highest CLIP similarity):")
    for img_id, score in ranked[-args.top_k :][::-1]:
        print(f"  {img_id:<10}  {score:.4f}")

    # ── save CSV ──
    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "clip_similarity"])
            for img_id, score in zip(ids, sims):
                writer.writerow([img_id, f"{score:.6f}"])
        print(f"\nPer-image results saved to: {args.save_csv}")

    # ── save JSON summary ──
    if args.save_json:
        summary = {
            "pairs": len(pairs),
            "clip_similarity": {"mean": mean_sim, "min": min_sim, "max": max_sim},
            "fid": fid_score,
            "clip_model": args.clip_model,
        }
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {args.save_json}")


if __name__ == "__main__":
    main()
