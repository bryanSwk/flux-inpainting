#!/usr/bin/env python3
"""
Batch background replacement using Flux1-Fill-dev (mflux / Apple Silicon).

Reads bg1k_info.txt to assign category-aware prompts, then processes every
image/mask pair in JD_1K/bg1k_imgs + JD_1K/bg1k_masks.

Usage:
    python batch_infer.py \
        --data-dir  JD_1K \
        --output-dir outputs \
        [--start 0] [--end 100] \
        [--steps 28] [--guidance 30] [--quantize 8] [--seed 42]

Outputs are written to:
    <output-dir>/<image_id>.png

Already-completed images are skipped automatically (resumable).
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ── Category → background prompt mapping ─────────────────────────────────────

CATEGORY_PROMPTS: dict[str, str] = {
    # Furniture / large appliances
    "refrigerator":        "modern kitchen interior, clean white walls, soft natural light",
    "dining table":        "bright Scandinavian dining room, wooden floor, natural window light",
    "fabric sofa":         "cosy living room interior, warm afternoon sunlight, neutral tones",
    "leather sofa":        "modern living room, light concrete walls, diffused daylight",
    "solid wood sofa":     "minimalist Japanese interior, tatami floor, soft ambient light",
    "leather bed":         "elegant bedroom, linen curtains, warm morning light",
    "fabric bed":          "bright bedroom interior, soft pastel tones, natural light",
    "sofa bed":            "contemporary living room, clean lines, warm neutral palette",
    "massage chair":       "modern wellness room, soft indirect lighting, neutral background",
    "computer chair":      "clean home office, light wooden desk, soft window light",
    "infant dining chair": "cheerful nursery room, pastel colours, soft diffused light",
    "bookshelf":           "minimalist study room, white walls, natural daylight",
    "computer desk":       "clean bright home office, soft window light",
    "latex mattress":      "bright bedroom, white bedding, airy natural light",

    # Kitchen / cooking appliances
    "electric pressure cooker": "modern kitchen counter, clean marble surface, bright kitchen light",
    "rice cooker":              "clean kitchen countertop, tile background, warm daylight",
    "wok":                      "bright professional kitchen counter, stainless steel surface",
    "electric baking pan":      "kitchen counter, bright overhead lighting, clean background",
    "induction cooker or electric ceramic cooker": "marble kitchen counter, modern kitchen, natural light",
    "coffee machine":           "cosy café counter, warm lighting, wood surface",
    "chef machine or dough mixer": "clean kitchen island, soft bright studio light",
    "wall breaking machine":    "modern kitchen counter, white tiles, bright natural light",
    "soup pot":                 "wooden kitchen table, rustic background, warm light",
    "multifunctional pot":      "clean kitchen surface, bright studio lighting",
    "health pot":               "wooden table, green plant nearby, warm natural light",
    "air fryer":                "bright kitchen counter, clean minimal background",
    "electric kettle":          "kitchen counter, morning light, clean bright setting",
    "hot pot":                  "restaurant setting, warm ambient lighting, dark wooden table",
    "meat grinder or vegetable cutting machine": "kitchen counter, clean bright background",
    "electric oven":            "bright modern kitchen, white tiles, soft light",

    # Watches
    "second hand watches":  "elegant dark display surface, jeweller's studio lighting, luxury feel",
    "national watch":       "polished marble surface, warm product photography lighting",
    "european and american watches": "premium dark velvet surface, dramatic side lighting",
    "japanese and korean watches":   "clean minimalist surface, soft diffused studio light",
    "swiss watch":          "luxury dark display background, jeweller's spotlight",
    "virtue watch":         "clean white display surface, soft studio lighting",
    "watch accessories":    "dark velvet display, warm accent lighting",

    # Small electronics
    "cell phone":           "clean white desk, minimal flat-lay, soft studio light",
    "laptop accessories":   "wooden desk, natural window light, clean background",
    "desktop":              "modern desk setup, clean white wall, bright office light",
    "monitor":              "clean office desk, soft ambient light",
    "tablet":               "white desk, flat-lay, bright minimalist studio",
    "keyboard":             "wooden desk, soft window light, clean setup",
    "mouse":                "clean desk surface, soft studio lighting",
    "mouse pad":            "wooden desk, soft diffused light",
    "graphics card":        "dark technical background, dramatic product lighting",
    "game notebook":        "dark gaming desk, RGB accent lighting, dramatic look",
    "notebook":             "wooden desk, natural light, clean background",
    "mobile power":         "clean white surface, soft product photography lighting",
    "printer":              "clean office setting, bright overhead light",
    "projector":            "dark home cinema setting, ambient warm glow",
    "speakers or audio":    "clean wooden shelf, soft warm light",
    "home theater":         "dark entertainment room, moody ambient lighting",
    "mp3 or mp4":           "clean white surface, bright product lighting",
    "bluetooth or wireless headphones": "dark premium surface, dramatic product lighting",
    "surveillance cameras": "clean neutral background, soft studio lighting",
    "sweeping robot":       "clean tile floor, bright home interior",
    "second-hand phone":    "clean white surface, bright product lighting",
    "mobile phone case or protective cover": "clean white desk, soft light",

    # Food & beverages
    "biscuits or puffed":   "rustic wooden table, natural daylight, food styling",
    "pastries or snacks":   "wooden serving board, bakery aesthetic, warm light",
    "pastry":               "marble table, elegant bakery setting, soft light",
    "chocolate":            "dark marble surface, dramatic side lighting",
    "candy":                "colourful flat-lay on bright pastel surface",
    "grains":               "rustic burlap surface, natural daylight, earthy tones",
    "beef":                 "butcher block wood, dramatic directional light",
    "chicken":              "clean white plate, bright food photography light",
    "fish":                 "crushed ice display, seafood market aesthetic, cool light",
    "mutton":               "wooden cutting board, warm rustic kitchen light",
    "hot pot":              "restaurant wooden table, warm ambient light",
    "instant food":         "clean kitchen counter, bright natural light",
    "dumplings or wontons": "bamboo steamer, warm steam, natural light",
    "seafood snacks":       "coastal wooden surface, natural beach light",
    "seafood dishes":       "elegant restaurant plate, dark background, spotlight",
    "delicatessen cured meat": "rustic wooden board, warm overhead light",
    "meat and poultry dishes": "elegant dining plate, warm restaurant light",
    "ice cream":            "bright pastel background, summer light, flat-lay",
    "dried tofu or vegetarian snacks": "clean neutral surface, soft natural light",

    # Tea & health products
    "longjing":             "elegant tea table, bamboo mat, soft natural light",
    "pu'er":                "dark wood tea table, warm atmospheric light",
    "tieguanyin":           "Chinese tea ceremony setting, ceramic background, soft light",
    "green tea":            "zen garden setting, bamboo, soft natural light",
    "black tea":            "cosy British afternoon tea setting, warm light",
    "white tea":            "clean white background, minimal tea styling, diffused light",
    "jasmine tea":          "floral white surface, natural spring light",
    "herbal tea":           "wooden rustic table, dried herbs, warm natural light",
    "complete tea set":     "traditional Chinese tea ceremony table, soft ambient light",
    "tea bar machine":      "modern tea shop counter, warm lighting",

    # Beauty & personal care
    "lotion or cream":      "clean white marble surface, soft product photography light",
    "toner or lotion":      "minimalist white studio, soft overhead light",
    "facial mask":          "spa setting, white marble, soft diffused light",
    "facial essence":       "luxury marble surface, elegant product lighting",
    "cleansing":            "clean bathroom counter, soft natural light",
    "shampoo":              "clean white bathroom surface, soft bright light",
    "body wash":            "spa bathroom setting, soft warm light",
    "perfume":              "luxury black velvet surface, dramatic directional light",
    "beauty device":        "clean white studio, soft product lighting",
    "sun protection":       "outdoor beach setting, soft bright daylight",

    # Clothing & accessories
    "bag":                  "clean studio background, soft fashion photography light",
    "women shoulder or crossbody bag": "elegant studio, soft neutral background, fashion light",
    "men backpack":         "outdoor urban setting, natural light",
    "business briefcase":   "office desk, professional warm lighting",
    "men wallet":           "leather desk surface, warm product lighting",
    "laptop bag":           "clean office setting, soft natural light",
    "suitcase":             "airport terminal, bright clean background",
    "men t-shirt":          "clean white studio, soft fashion photography light",
    "men jackets":          "urban outdoor setting, cool natural daylight",
    "men casual shoes":     "clean studio floor, soft side light",
    "sports and casual shoes": "clean studio, bright product photography light",
    "running shoes":        "athletic track background, action-inspired lighting",
    "men slippers":         "cosy home interior, warm soft light",
    "sports slippers":      "gym floor, bright overhead light",
    "casual pants":         "clean studio background, soft diffused light",
    "casual socks":         "clean wooden floor, bright flat-lay light",
    "pajamas or lounge clothes": "cosy bedroom setting, warm soft light",
    "sun hat":              "beach background, bright natural sunlight",
    "men underwear":        "clean white studio, minimal fashion lighting",
    "plus size women clothing": "bright clean studio, soft fashion light",
    "vest or vest":         "clean light studio, soft natural light",
    "shirt":                "clean white hanger, soft neutral light",

    # Jewellery
    "necklace":             "dark velvet display, warm jewellery lighting",
    "bracelet or anklet":   "polished marble surface, elegant product lighting",
    "earring":              "dark display background, jewellery spotlight",
    "jade pendant":         "natural stone surface, warm side lighting",
    "silver bracelet":      "dark velvet background, dramatic jewellery lighting",

    # Health / supplements
    "coenzyme q10":         "clean white pharmaceutical surface, bright studio light",
    "dried bird nest":      "ceramic bowl, traditional Chinese setting, warm light",
    "ready to eat bird nest": "elegant white ceramic, soft warm light",
    "dried sea cucumber":   "natural dark stone surface, earthy warm light",
    "cordyceps sinensis":   "wooden tray, traditional setting, warm natural light",
    "wolfberry":            "wooden bowl, natural rustic background, warm light",
    "lotion or cream":      "white marble surface, clean product photography",
    "nourish the liver or clear the lungs": "herbal clean background, soft natural light",
    "food and medicine come from the same source": "natural wooden surface, warm light",

    # Home & furniture
    "decorative ornaments": "bright minimalist shelf, neutral wall, soft ambient light",
    "multifunctional storage rack": "bright clean home interior, natural light",
    "storage box":          "clean organised room, soft overhead light",
    "ceiling lamp":         "bright white ceiling installation, clean interior",
    "mood lighting":        "dark cosy room, warm atmospheric glow",
    "scented candle":       "dark atmospheric setting, warm candlelight glow",
    "everlasting flower":   "bright minimal background, soft natural light",
    "music box":            "wooden surface, warm vintage light",
    "safe or box":          "clean neutral background, professional product lighting",
    "dish rack":            "bright kitchen setting, clean natural light",
    "draw paper":           "clean white desk, flat-lay, soft light",
    "this booklet or notes": "wooden desk, natural light, stationery flat-lay",

    # Appliances / home
    "washing machine":      "clean laundry room, bright white tiles, overhead light",
    "dryer":                "clean laundry room, bright background",
    "clothes dryer":        "bright bathroom or laundry, clean background",
    "air conditioner":      "clean modern living room, white walls, natural light",
    "air purifier":         "clean bright living room, modern interior",
    "water purifier":       "clean kitchen counter, bright lighting",
    "smart toilet":         "modern bathroom, clean white tiles, bright light",
    "bathroom cabinet":     "modern bathroom, clean white background",
    "sink":                 "modern kitchen or bathroom, clean surfaces, bright light",
    "toilet":               "clean modern bathroom, white tiles",
    "gas water heater":     "clean bathroom wall, bright bathroom light",
    "electric heater":      "cosy living room corner, warm ambient light",
    "bath heater":          "clean bathroom, bright natural light",

    # Food storage & tableware
    "ceramic or mug":       "wooden café table, warm coffee shop light",
    "cup":                  "clean kitchen surface, bright natural light",
    "bowl":                 "wooden dining table, warm food styling light",
    "glass":                "clean bar or kitchen surface, bright light",
    "plate or dish":        "elegant restaurant setting, warm directional light",
    "cutlery set":          "elegant dark dining table, warm restaurant light",
    "tableware and water utensils": "dining table, warm ambient light",
    "thermos cup":          "outdoor adventure setting, natural light",
    "thermos kettle":       "kitchen counter, warm morning light",
    "sport bottle":         "outdoor mountain or gym, bright active light",
    "plastic cup":          "clean white studio, soft bright light",
    "wine":                 "wine cellar or dark elegant bar, warm mood lighting",
    "liquor":               "premium dark bar counter, dramatic spotlight",
    "wine cabinet":         "elegant dining room, warm ambient light",

    # Miscellaneous
    "building blocks":      "bright children's playroom, colourful natural light",
    "early childhood education": "bright nursery, natural daylight, cheerful tones",
    "baby bottle nipple":   "clean white soft nursery background",
    "infant milk powder":   "clean baby product background, soft light",
    "children milk powder": "bright cheerful nursery, soft pastel background",
    "basketball":           "sports court, dramatic overhead light",
    "motorcycle helmet":    "clean studio, dramatic product lighting",
    "car perfume":          "clean car interior, bright modern light",
    "lighter":              "dark moody background, warm flame light",
    "outdoor tools":        "outdoor wilderness setting, natural rugged light",
    "rolling paper":        "clean neutral background, soft studio light",
    "switch socket":        "clean white wall, bright interior light",
    "broken wall or refined pieces": "construction site, natural industrial light",
    "chassis":              "dark premium tech background, dramatic product lighting",
    "projector":            "dark cinema room, ambient glow",
    "fish tank or aquarium": "clean aquarium shop, subtle blue ambient light",
    "seat cushion":         "bright living room, natural diffused light",
    "pillow cushion":       "cosy bright bedroom, soft natural light",
    "trend blind box":      "playful colourful background, soft studio light",
    "candied dried fruits": "rustic wooden surface, warm natural light",
    "roasted nuts":         "rustic wooden bowl, warm food photography light",
    "dried mushrooms":      "wooden cutting board, natural earthy light",
    "dried meat dried meat": "rustic wooden surface, warm directional light",
    "investment funds":     "clean professional desk, bright business setting",
    "investment collection": "elegant dark display, warm accent lighting",
    "accessories":          "clean neutral surface, soft product lighting",
}

DEFAULT_PROMPT = "clean white studio background, soft diffused product photography light"


def get_prompt(category: str) -> str:
    return CATEGORY_PROMPTS.get(category.lower(), DEFAULT_PROMPT)


# ── mask conversion ───────────────────────────────────────────────────────────

def make_fill_mask(mask_path: str, tmp_dir: str, idx: int) -> str:
    """Convert BG-1K RGBA mask → binary fill mask (white=fill, black=keep)."""
    mask = Image.open(mask_path).convert("RGBA")
    alpha = np.array(mask)[:, :, 3]
    fill = np.where(alpha > 128, 0, 255).astype(np.uint8)
    fill_img = Image.fromarray(fill, mode="L")
    out_path = os.path.join(tmp_dir, f"fill_{idx}.png")
    fill_img.save(out_path)
    return out_path


def resize_to_multiple(image_path: str, tmp_dir: str, idx: int, multiple: int = 64) -> tuple[str, tuple[int, int]]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
        out_path = os.path.join(tmp_dir, f"resized_{idx}.png")
        img.save(out_path)
        return out_path, (new_w, new_h)
    return image_path, (w, h)


# ── data loading ──────────────────────────────────────────────────────────────

def load_info(data_dir: str) -> dict[str, str]:
    """Return {image_id: category} from bg1k_info.txt."""
    info_path = os.path.join(data_dir, "bg1k_info.txt")
    mapping: dict[str, str] = {}
    with open(info_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                img_id = parts[0].replace(".png", "").replace(".jpg", "")
                mapping[img_id] = parts[1].strip()
    return mapping


def collect_pairs(data_dir: str) -> list[tuple[str, str, str]]:
    """Return list of (image_id, image_path, mask_path) sorted by numeric id."""
    imgs_dir = os.path.join(data_dir, "bg1k_imgs")
    masks_dir = os.path.join(data_dir, "bg1k_masks")
    pairs = []
    for fname in os.listdir(imgs_dir):
        if not fname.endswith((".png", ".jpg", ".jpeg")):
            continue
        img_id = fname.rsplit(".", 1)[0]
        mask_fname = f"{img_id}_mask.png"
        mask_path = os.path.join(masks_dir, mask_fname)
        if os.path.exists(mask_path):
            pairs.append((img_id, os.path.join(imgs_dir, fname), mask_path))
    pairs.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    return pairs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch Flux1-Fill background replacement on BG-1K")
    parser.add_argument("--data-dir",   default="JD_1K",   help="Path to JD_1K directory (default: JD_1K)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--start",    type=int, default=0,    help="Start index (inclusive, default: 0)")
    parser.add_argument("--end",      type=int, default=None, help="End index (exclusive, default: all)")
    parser.add_argument("--steps",    type=int,   default=28,   help="Inference steps (default: 28)")
    parser.add_argument("--guidance", type=float, default=30.0, help="Guidance scale (default: 30)")
    parser.add_argument("--quantize", type=int,   default=8, choices=[4, 6, 8],
                        help="Weight quantization bits (default: 8)")
    parser.add_argument("--seed",     type=int,   default=42,   help="RNG seed (default: 42)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print what would be processed without running inference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    info = load_info(args.data_dir)
    pairs = collect_pairs(args.data_dir)
    pairs = pairs[args.start: args.end]

    if args.dry_run:
        for img_id, img_path, mask_path in pairs:
            category = info.get(img_id, "unknown")
            prompt = get_prompt(category)
            out = os.path.join(args.output_dir, f"{img_id}.png")
            print(f"[{img_id}] category={category!r}")
            print(f"         prompt  ={prompt!r}")
            print(f"         output  ={out}")
        return

    # Load model once
    from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
    print(f"Loading Flux1-Fill-dev (quantize={args.quantize})...")
    flux = Flux1Fill(quantize=args.quantize)

    with tempfile.TemporaryDirectory() as tmp:
        for img_id, img_path, mask_path in tqdm(pairs, desc="Generating"):
            out_path = os.path.join(args.output_dir, f"{img_id}.png")
            if os.path.exists(out_path):
                tqdm.write(f"[skip] {img_id} already done")
                continue

            category = info.get(img_id, "unknown")
            prompt = get_prompt(category)

            fill_mask_path = make_fill_mask(mask_path, tmp, img_id)
            resized_path, (width, height) = resize_to_multiple(img_path, tmp, img_id)

            try:
                image = flux.generate_image(
                    seed=args.seed,
                    prompt=prompt,
                    width=width,
                    height=height,
                    guidance=args.guidance,
                    image_path=resized_path,
                    num_inference_steps=args.steps,
                    masked_image_path=fill_mask_path,
                )
                image.save(path=out_path)
                tqdm.write(f"[done] {img_id} → {out_path}")
            except Exception as exc:
                tqdm.write(f"[error] {img_id}: {exc}")


if __name__ == "__main__":
    main()
