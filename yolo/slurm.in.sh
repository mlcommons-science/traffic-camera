#!/bin/bash
#SBATCH --job-name=cc_{ee.identifier}
#SBATCH --output=darknet_train.out
#SBATCH --error=darknet_train.err
{slurm.sbatch}

set -euo pipefail

# Apptainer isn't on compute nodes unless you load it
module load apptainer || true

echo "Running on $(hostname)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# We are in yolo/project/<identifier>, so yolo dir is two levels up
YOLO_DIR="$(cd "$SUBMIT_DIR/../.." && pwd)"
SIF="$YOLO_DIR/my_darknet_container.sif"

WORKSPACE="$SUBMIT_DIR/workspace"
OUTPUTS="$SUBMIT_DIR/outputs"
HOME_DIR="$SUBMIT_DIR/home"

mkdir -p "$WORKSPACE" "$OUTPUTS" "$HOME_DIR" \
         "$WORKSPACE/.cache" "$WORKSPACE/.mplcache"

echo "YOLO_DIR=$YOLO_DIR"
echo "Using SIF: $SIF"
ls -lh "$SIF"

# Run bash inside the container so pipefail works,
# build darknet into /workspace/darknet, then train.
apptainer exec --nv --fakeroot \
  --home "$HOME_DIR":/home/$USER \
  -B "$WORKSPACE":/workspace:rw \
  -B "$OUTPUTS":/outputs:rw \
  --env DARKNET_PARENT=/workspace \
  --env MPLCONFIGDIR=/workspace/.mplcache \
  --env XDG_CACHE_HOME=/workspace/.cache \
  "$SIF" \
  bash -lc "
    set -euo pipefail
    build_darknet.sh
    python -u -m chocolatechip.model_training.train \
      --profile {experiment.profile} \
      --template {experiment.template} \
      --val-frac {experiment.val_frac} \
      --num-gpus {experiment.num_gpus} \
      --color-preset {experiment.color_preset} \
      --ultra-model {experiment.ultra_model} \
      --no-sweep
  "

