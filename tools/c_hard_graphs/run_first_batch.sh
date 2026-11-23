#!/usr/bin/env bash
# Run from the repository root. Each worker writes only the listed model/stage pairs.
set -euo pipefail
job=${1:?specify original, trained, or report}
mode=${2:?specify smoke or formal}
extra=()
case "$mode" in smoke) extra+=(--smoke);; formal) ;; *) exit 2;; esac
py=${PYTHON:-python}
dir=tools/c_hard_graphs
out=${FIRST_BATCH_OUT:-tmp/analysis/first-batch-20261123}
run() {
    "$py" "$dir/first_batch.py" --data-root "${DATA_ROOT:?training worktree required}" --out "$out" "${extra[@]}" "$@"
}
case "$job" in
    original)
        run --models C_s42 C_s1234 --stages natural history fanout
        run --models A_s42 B_s42 --stages fanout
        ;;
    trained)
        run --models C_s42-update C_s1234-update --stages mark
        for epoch in 10 30; do
            run --models C_s42-update C_s1234-update --stages mark --mark-epoch "$epoch"
        done
        run --models C_s42-correction C_s1234-correction C_s42-update C_s1234-update --stages standard layered expanded fanout
        ;;
    report)
        "$py" "$dir/first_batch_report.py" --out "$out" "${extra[@]}"
        ;;
    *) exit 2;;
esac
