#!/usr/bin/env bash
# Rerun P1 (4->5 / EMPTY->5) on the ORIGINAL P1 dataset.
# The old P1 files used Tr32Test128 N32 (82 eligible u), not heldout-seed42.
# Usage: bash rerun_p1_oldset_20261113.sh [gpu_index]   (default 7)
set +e
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
GPU=${1:-7}
REPO=/workspace/lab-xj/users/jinxi/code/yhy/algorithmic-generalization-transformer-architectures
PY=/workspace/lab-xj/users/jinxi/.conda/envs/yhy11.6/bin/python
DS="$REPO/experiments/evaluation/val_datasets/Tr32Test128/ADD/testdata_N32.pt"
OUT="$REPO/tmp/analysis/parent-2x2-fixed/P1_oldset"
LOG="$REPO/tmp/analysis/rerun-fixed-20261113/p1_oldset"
mkdir -p "$OUT" "$LOG"
cd "$REPO/experiments/evaluation" || exit 1
B_CFG="B-M6-Nodes32-ADD-L4H16D384-DeBERTa"
B_GRP="Nodes32-ADD - L4H16D384_DeBERTa_recorrection"
B_BASE="$REPO/tmp/runs/B-M6/checkpoints"

run () {
  local tag=$1 run=$2 base=$3 ckpt=$4 timing=$5
  echo "[$(date '+%T')] START $tag"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$REPO/tmp/scripts/parent_corruption_2x2.py" \
    --config_dir_name "$B_CFG" --group_name "$B_GRP" --run_name "$run" \
    --ckpt_base_dir "$base" --ckpt_name "$ckpt" \
    --ds_file "$DS" --n_graphs 128 --batch_size 8 --k 4 \
    --start teacher --timing "$timing" --value_mode four_to_five --extra_steps 3 \
    --tag "$tag" --out_dir "$OUT" > "$LOG/$tag.log" 2>&1
  echo "[$(date '+%T')] END $tag rc=$?"
}

run P1_BM6_s42_4to5_128        "seed-42 - 2026-11-07-17:47:37" "$B_BASE" last.ckpt after_u
run P1_cyhB_s42_4to5_128       "seed-42 - 2026-11-08-10:42:15" "$REPO/tmp/cyh_ckpt/seed-42" "epoch=84-val_loss=0.0000.ckpt" after_u
run P1_cyhB_s1234_4to5_128     "seed-1234 - 2026-11-08-10:42:15" "$REPO/tmp/cyh_ckpt/seed-1234" "epoch=88-val_loss=0.0000.ckpt" after_u
run P1_cyhB_s2025_4to5_128     "seed-2025 - 2026-11-08-11:02:06" "$REPO/tmp/cyh_ckpt/seed-2025" "epoch=74-val_loss=0.0000.ckpt" after_u
run P1_BM6_s42_EMPTY5_128      "seed-42 - 2026-11-07-17:47:37" "$B_BASE" last.ckpt before_u
run P1_cyhB_s42_EMPTY5_128     "seed-42 - 2026-11-08-10:42:15" "$REPO/tmp/cyh_ckpt/seed-42" "epoch=84-val_loss=0.0000.ckpt" before_u
run P1_cyhB_s1234_EMPTY5_128   "seed-1234 - 2026-11-08-10:42:15" "$REPO/tmp/cyh_ckpt/seed-1234" "epoch=88-val_loss=0.0000.ckpt" before_u
run P1_cyhB_s2025_EMPTY5_128   "seed-2025 - 2026-11-08-11:02:06" "$REPO/tmp/cyh_ckpt/seed-2025" "epoch=74-val_loss=0.0000.ckpt" before_u
echo P1_OLDSET_DONE