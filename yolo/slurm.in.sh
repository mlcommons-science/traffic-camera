#!/bin/bash
#SBATCH --job-name=cc_{ee.identifier}
#SBATCH --output=darknet_train.out
#SBATCH --error=darknet_train.err
{slurm.sbatch}

set -euo pipefail

echo "Running on $(hostname)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Ensure module command exists in batch jobs
source /etc/profile.d/modules.sh || true
module load apptainer

# Always start from the submit dir (EE makes this per-experiment)
cd "$SLURM_SUBMIT_DIR"

# Locate YOLO dir + SIF (relative, no usernames, no git assumptions)
YOLO_DIR="$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)"
SIF="$YOLO_DIR/my_darknet_container.sif"

echo "YOLO_DIR=$YOLO_DIR"
echo "Using SIF: $SIF"

if [ ! -f "$SIF" ]; then
  echo "FATAL: SIF not found at $SIF"
  echo "Looked relative to SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
  exit 2
fi

# Writable per-experiment dirs
EXP_DIR="$SLURM_SUBMIT_DIR"
HOST_WS="$EXP_DIR/host_workspace"
WORKSPACE="$EXP_DIR/workspace"
OUTPUTS="$EXP_DIR/outputs"
HOME_DIR="$EXP_DIR/home"
MPLDIR="$EXP_DIR/mplconfig"

mkdir -p "$HOST_WS" "$WORKSPACE" "$OUTPUTS" "$HOME_DIR" "$MPLDIR"

# Keep python/matplotlib caches off read-only $HOME
export MPLCONFIGDIR="$MPLDIR"
export XDG_CACHE_HOME="$WORKSPACE/.cache"
export TMPDIR="$WORKSPACE/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"

# Use allocated CPUs for any threaded libs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Build Darknet into /host_workspace so chocolatechip can find it
export DARKNET_PARENT="/host_workspace"

# ---- color preset logic ----
# If color_preset is null/None, omit the flag entirely.
COLOR_PRESET="{experiment.color_preset}"
COLOR_ARGS=""
if [[ -n "$COLOR_PRESET" && "$COLOR_PRESET" != "None" && "$COLOR_PRESET" != "null" ]]; then
  COLOR_ARGS="--color-preset $COLOR_PRESET"
fi


# Run training inside container with writable binds
apptainer exec --nv --fakeroot \
  --home "$HOME_DIR" \
  -B "$HOST_WS":/host_workspace:rw \
  -B "$WORKSPACE":/workspace:rw \
  -B "$OUTPUTS":/outputs:rw \
  -B "$MPLDIR":/mplconfig:rw \
  --env DARKNET_PARENT=/host_workspace \
  --env MPLCONFIGDIR=/mplconfig \
  "$SIF" \
  bash -lc "
    set -euo pipefail
    /usr/local/bin/build_darknet.sh
    python -u -m chocolatechip.model_training.train \
      --profile {experiment.profile} \
      --template {experiment.template} \
      --val-frac {experiment.val_frac} \
      --num-gpus {experiment.num_gpus} \
      $COLOR_ARGS \
      --ultra-model {experiment.ultra_model} \
      --no-sweep
  "

