# 1. Setup
./setup.sh
source .venv/bin/activate

# 2. HuggingFace login
hf auth login

# 3. Single image
python infer.py \
  --image JD_1K/bg1k_imgs/0.png \
  --mask  JD_1K/bg1k_masks/0_mask.png \
  --prompt "modern kitchen interior, clean white walls, soft natural light" \
  --output result.png

# 4. Batch (all 1000 images, resumable)
python batch_infer.py --data-dir JD_1K --output-dir outputs