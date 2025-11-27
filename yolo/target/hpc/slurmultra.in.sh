#!/bin/bash
#SBATCH --job-name=cc_{ee.identifier}
#SBATCH --output=ultra_train.out
#SBATCH --error=ultra_train.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=1-00:00:00
{slurm.sbatch}

set -euo pipefail

echo "Running on $(hostname)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# Ensure module command exists in batch jobs
source /etc/profile.d/modules.sh || true
module load apptainer

# Always start from the submit dir (EE makes this per-experiment)
cd "$SLURM_SUBMIT_DIR"

# Locate ULTRALYTICS dir + SIF (relative, no usernames, no git assumptions)
ULTRA_DIR="$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)"
SIF="$ULTRA_DIR/my_ultralytics_container.sif"

echo "ULTRA_DIR=$ULTRA_DIR"
echo "Using SIF: $SIF"

if [ ! -f "$SIF" ]; then
  echo "FATAL: SIF not found at $SIF"
  echo "Looked relative to SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
  exit 2
fi

# Writable per-experiment dirs
EXP_DIR="$SLURM_SUBMIT_DIR"
WORKSPACE="$EXP_DIR/workspace"
OUTPUTS="$EXP_DIR/outputs"
HOME_DIR="$EXP_DIR/home"
MPLDIR="$EXP_DIR/mplconfig"
CACHEDIR="$WORKSPACE/.cache"
CONFIGDIR="$WORKSPACE/.config"

#TMPDIR_HOST="/scratch/$USER/tmp_${SLURM_JOB_ID}"
TMPDIR_HOST={system.tmpdir_host}
mkdir -p "$TMPDIR_HOST"

mkdir -p "$WORKSPACE" "$OUTPUTS" "$HOME_DIR" "$MPLDIR" "$CACHEDIR" "$CONFIGDIR" "$TMPDIR_HOST"

# Keep python/matplotlib/ultralytics caches off read-only $HOME
export MPLCONFIGDIR="$MPLDIR"
export XDG_CACHE_HOME="$CACHEDIR"
export XDG_CONFIG_HOME="$CONFIGDIR"

# Use allocated CPUs for any threaded libs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ---- color preset logic (if your ultra profiles read it) ----
COLOR_PRESET="{experiment.color_preset}"
COLOR_ARGS=""
if [[ -n "$COLOR_PRESET" && "$COLOR_PRESET" != "None" && "$COLOR_PRESET" != "null" ]]; then
  COLOR_ARGS="--color-preset $COLOR_PRESET"
fi

# ---- dataset root logic ----
DATASET_ROOT="{system.dataset_root}"
DATASET_ARGS=""
if [[ -n "$DATASET_ROOT" && "$DATASET_ROOT" != "None" && "$DATASET_ROOT" != "null" ]]; then
  DATASET_ARGS="--dataset-root $DATASET_ROOT"
fi

# cap dataloader workers to allocated CPUs (or lower)
export ULTRA_WORKERS=4

# Run training inside container with writable binds
apptainer exec --nv --fakeroot \
  --home "$HOME_DIR" \
  -B "$WORKSPACE":/workspace:rw \
  -B "$EXP_DIR":/workspace/project:rw \
  -B "$OUTPUTS":/outputs:rw \
  -B "$MPLDIR":/mplconfig:rw \
  -B "$TMPDIR_HOST":/tmp:rw \
  --env MPLCONFIGDIR=/mplconfig \
  --env XDG_CACHE_HOME=/workspace/.cache \
  --env XDG_CONFIG_HOME=/workspace/.config \
  --env DATA_ROOT=/workspace/.cache/datasets \
  --env ULTRA_WORKERS="$ULTRA_WORKERS" \
  --env TMPDIR=/tmp \
  "$SIF" \
  bash -lc "
    set -euo pipefail

    python -u -m chocolatechip.model_training.train \
      --profile {experiment.profile} \
      --val-frac {experiment.val_frac} \
      --num-gpus {experiment.num_gpus} \
      $DATASET_ARGS \
      --ultra-model {experiment.ultra_model} \
      --no-sweep
  "

