#!/usr/bin/env bash
# Rerun all analysis affected by the 2026-11-13 fixes.
# Usage: bash rerun_fixed_20261113.sh <gpu_index>
# GPUs used: 2/3/5/6/7 (do NOT use 0/1/4: C0 training).
set +e
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
GPU=${1:?gpu index required}
REPO=/workspace/lab-xj/users/jinxi/code/yhy/algorithmic-generalization-transformer-architectures
PY=/workspace/lab-xj/users/jinxi/.conda/envs/yhy11.6/bin/python
PFIX="$REPO/tmp/analysis/parent-2x2-fixed"
VFIX="$REPO/tmp/analysis/inject-v2-fixed"
LOGS="$REPO/tmp/analysis/rerun-fixed-20261113"
mkdir -p "$PFIX" "$VFIX" "$LOGS"
cd "$REPO/experiments/evaluation" || exit 1

DS32="$REPO/tmp/analysis/heldout-seed42/ADD/testdata_N32.pt"
DS128="$REPO/tmp/analysis/heldout-seed42/ADD/testdata_N128.pt"

A_CFG="A-L4D384fast-Nodes32-ADD-L4H16D384-DeBERTa"
A_GRP="Nodes32-ADD - L4H16D384_DeBERTa"
A_BASE="$REPO/tmp/runs/A-L4D384fast/checkpoints"
B_CFG="B-M6-Nodes32-ADD-L4H16D384-DeBERTa"
B_GRP="Nodes32-ADD - L4H16D384_DeBERTa_recorrection"
B_BASE="$REPO/tmp/runs/B-M6/checkpoints"

run_parent () {
  local tag=$1 cfg=$2 group=$3 run=$4 base=$5 ckpt=$6 ds=$7 n=$8 timing=$9 start=${10} mode=${11} outdir=${12}
  echo "[$(date '+%F %T')] START $tag (gpu=$GPU n=$n $start/$timing/$mode)"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$REPO/tmp/scripts/parent_corruption_2x2.py" \
    --config_dir_name "$cfg" --group_name "$group" --run_name "$run" \
    --ckpt_base_dir "$base" --ckpt_name "$ckpt" \
    --ds_file "$ds" --n_graphs "$n" --batch_size 8 --k 4 \
    --start "$start" --timing "$timing" --value_mode "$mode" --extra_steps 3 \
    --tag "$tag" --out_dir "$outdir" > "$LOGS/gpu${GPU}_${tag}.log" 2>&1
  echo "[$(date '+%F %T')] END $tag rc=$?"
}

run_v2 () {
  local tag=$1 cfg=$2 group=$3 run=$4 base=$5 ckpt=$6 ds=$7 n=$8 k=$9
  echo "[$(date '+%F %T')] START $tag (gpu=$GPU n=$n k=$k)"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$REPO/tmp/scripts/inject_value_discrete_v2.py" \
    --config_dir_name "$cfg" --group_name "$group" --run_name "$run" \
    --ckpt_base_dir "$base" --ckpt_name "$ckpt" \
    --ds_file "$ds" --n_graphs "$n" --batch_size 8 --start teacher --k "$k" \
    --value_mode a_plus_1 --extra_steps 3 --tag "$tag" \
    --out_dir "$VFIX" > "$LOGS/gpu${GPU}_${tag}.log" 2>&1
  echo "[$(date '+%F %T')] END $tag rc=$?"
}

# A' / B / cyh weights
A42=( "$A_CFG" "$A_GRP" "seed-42 - 2026-11-09-15:42:45" "$A_BASE" "epoch=86-val_loss=0.0000.ckpt" )
A1234=( "$A_CFG" "$A_GRP" "seed-1234 - 2026-11-09-16:15:53" "$A_BASE" "epoch=90-val_loss=0.0000.ckpt" )
A2025=( "$A_CFG" "$A_GRP" "seed-2025 - 2026-11-09-16:30:43" "$A_BASE" "epoch=94-val_loss=0.0000.ckpt" )
B42=( "$B_CFG" "$B_GRP" "seed-42 - 2026-11-07-17:47:37" "$B_BASE" "last.ckpt" )
C42=( "$B_CFG" "$B_GRP" "seed-42 - 2026-11-08-10:42:15" "$REPO/tmp/cyh_ckpt/seed-42" "epoch=84-val_loss=0.0000.ckpt" )
C1234=( "$B_CFG" "$B_GRP" "seed-1234 - 2026-11-08-10:42:15" "$REPO/tmp/cyh_ckpt/seed-1234" "epoch=88-val_loss=0.0000.ckpt" )
C2025=( "$B_CFG" "$B_GRP" "seed-2025 - 2026-11-08-11:02:06" "$REPO/tmp/cyh_ckpt/seed-2025" "epoch=74-val_loss=0.0000.ckpt" )

parent () { run_parent "$1" "${@:2}"; }

case "$GPU" in
  2)
    # teacher after a_plus_1, 256 graphs, k=4 (5 runs)
    parent D2x2_Aprime_s42_N32_K4_ap1_after   "${A42[@]}"   "$DS32" 256 after_u teacher a_plus_1 "$PFIX"
    parent D2x2_BM6_s42_N32_K4_ap1_after     "${B42[@]}"   "$DS32" 256 after_u teacher a_plus_1 "$PFIX"
    parent D2x2_cyhB_s42_N32_K4_ap1_after    "${C42[@]}"   "$DS32" 256 after_u teacher a_plus_1 "$PFIX"
    parent D2x2_cyhB_s1234_N32_K4_ap1_after  "${C1234[@]}" "$DS32" 256 after_u teacher a_plus_1 "$PFIX"
    parent D2x2_cyhB_s2025_N32_K4_ap1_after  "${C2025[@]}" "$DS32" 256 after_u teacher a_plus_1 "$PFIX"
    # teacher before a_plus_1 (2 runs)
    parent D2x2_Aprime_s42_N32_K4_ap1_before "${A42[@]}" "$DS32" 256 before_u teacher a_plus_1 "$PFIX"
    parent D2x2_BM6_s42_N32_K4_ap1_before    "${B42[@]}" "$DS32" 256 before_u teacher a_plus_1 "$PFIX"
    ;;
  3)
    # own-state after/before a_plus_1 (4 runs)
    parent D2x2_own_Aprime_s42_N32_K4_ap1_after  "${A42[@]}" "$DS32" 256 after_u own a_plus_1 "$PFIX"
    parent D2x2_own_BM6_s42_N32_K4_ap1_after    "${B42[@]}" "$DS32" 256 after_u own a_plus_1 "$PFIX"
    parent D2x2_own_Aprime_s42_N32_K4_ap1_before "${A42[@]}" "$DS32" 256 before_u own a_plus_1 "$PFIX"
    parent D2x2_own_BM6_s42_N32_K4_ap1_before   "${B42[@]}" "$DS32" 256 before_u own a_plus_1 "$PFIX"
    # P1 4->5 and EMPTY->5, 128 graphs, k=4 (8 runs)
    mkdir -p "$PFIX/P1"
    parent P1_BM6_s42_4to5_128        "${B42[@]}"   "$DS32" 128 after_u teacher four_to_five "$PFIX/P1"
    parent P1_cyhB_s42_4to5_128       "${C42[@]}"   "$DS32" 128 after_u teacher four_to_five "$PFIX/P1"
    parent P1_cyhB_s1234_4to5_128     "${C1234[@]}" "$DS32" 128 after_u teacher four_to_five "$PFIX/P1"
    parent P1_cyhB_s2025_4to5_128     "${C2025[@]}" "$DS32" 128 after_u teacher four_to_five "$PFIX/P1"
    parent P1_BM6_s42_EMPTY5_128      "${B42[@]}"   "$DS32" 128 before_u teacher four_to_five "$PFIX/P1"
    parent P1_cyhB_s42_EMPTY5_128     "${C42[@]}"   "$DS32" 128 before_u teacher four_to_five "$PFIX/P1"
    parent P1_cyhB_s1234_EMPTY5_128   "${C1234[@]}" "$DS32" 128 before_u teacher four_to_five "$PFIX/P1"
    parent P1_cyhB_s2025_EMPTY5_128   "${C2025[@]}" "$DS32" 128 before_u teacher four_to_five "$PFIX/P1"
    ;;
  5)
    # v2 teacher N32 k=4, 256 graphs (7 runs)
    run_v2 Aprime_s42_newN32_k4   "${A42[@]}"   "$DS32" 256 4
    run_v2 BM6_s42_newN32_k4      "${B42[@]}"   "$DS32" 256 4
    run_v2 cyhB_s42_newN32_k4     "${C42[@]}"   "$DS32" 256 4
    run_v2 Aprime_s1234_newN32_k4 "${A1234[@]}" "$DS32" 256 4
    run_v2 cyhB_s1234_newN32_k4   "${C1234[@]}" "$DS32" 256 4
    run_v2 Aprime_s2025_newN32_k4 "${A2025[@]}" "$DS32" 256 4
    run_v2 cyhB_s2025_newN32_k4   "${C2025[@]}" "$DS32" 256 4
    ;;
  6)
    # v2 teacher N32 k=0 (4 runs)
    run_v2 Aprime_s42_newN32_kall   "${A42[@]}"   "$DS32" 256 0
    run_v2 BM6_s42_newN32_kall      "${B42[@]}"   "$DS32" 256 0
    run_v2 Aprime_s1234_newN32_kall "${A1234[@]}" "$DS32" 256 0
    run_v2 cyhB_s1234_newN32_kall   "${C1234[@]}" "$DS32" 256 0
    # v2 teacher N128 k=4 (2 of 6)
    run_v2 Aprime_s42_newN128_k4    "${A42[@]}"   "$DS128" 256 4
    run_v2 BM6_s42_newN128_k4       "${B42[@]}"   "$DS128" 256 4
    ;;
  7)
    # v2 teacher N128 k=4 (remaining 4)
    run_v2 Aprime_s1234_newN128_k4 "${A1234[@]}" "$DS128" 256 4
    run_v2 cyhB_s1234_newN128_k4   "${C1234[@]}" "$DS128" 256 4
    run_v2 Aprime_s2025_newN128_k4 "${A2025[@]}" "$DS128" 256 4
    run_v2 cyhB_s2025_newN128_k4   "${C2025[@]}" "$DS128" 256 4
    ;;
esac
echo "[$(date '+%F %T')] GPU$GPU QUEUE DONE"