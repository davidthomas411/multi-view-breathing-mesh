#!/bin/bash
# Skin-only dense tracking for one trial, 5 cameras in parallel, seeds on the markerless multi-view mesh.
# usage: run_lk_trial.sh TRIAL PTS OUTSUF      e.g.  run_lk_trial.sh ADB points_dense_wide _wide
cd "$(dirname "$0")"
for c in 1 2 3 4 5; do
  TRIAL=$1 PTS=$2 OUTSUF=$3 ${PY:-python} chest_lk_masked.py $c mv:0.5:41:33 mv:0.5:41:33:1:1e-6 > ../tmp/lk_$1_T$c.log 2>&1 &
done
wait
echo "LK_DONE $1 $2 $3"
