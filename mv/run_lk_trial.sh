#!/bin/bash
# Skin-only dense tracking for one trial, 5 cameras in parallel.
# usage: run_lk_trial.sh TRIAL PTS OUTSUF [SEED [EXTRA]]
#   SEED  = where the seed points are placed on the mesh: mv (markerless multi-view mesh, default), sap (Sapiens2 cold-start mesh), sam3d, fit (marker-fitted mesh, uses Vicon)
#   EXTRA = additional combination, e.g. mv:0.5:41:33:1:1e-6 (lower minimum eigenvalue)
# example: run_lk_trial.sh ADB points_dense_wide _wide
cd "$(dirname "$0")"
PY="${PY:-${PY:-python}}"; SEED="${4:-mv}"
for c in 1 2 3 4 5; do
  TRIAL=$1 PTS=$2 OUTSUF=$3 $PY chest_lk_masked.py $c $SEED:0.5:41:33 $5 > ../tmp/lk_$1_${SEED}_T$c.log 2>&1 &
done
wait
echo "LK_DONE $1 $2 $3 $SEED"
