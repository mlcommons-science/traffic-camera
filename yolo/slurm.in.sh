#!/bin/bash
#SBATCH --job-name=cc_{ee.identifier}
#SBATCH --output=darknet_train.out
#SBATCH --error=darknet_train.err
{slurm.sbatch}

set -euo pipefail
echo "Running on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# activate your env / container as you already do
cd {cloudmesh.cwd}   # or hardcode your repo path

# If you're using apptainer:
# apptainer run --nv ... my_darknet_container.sif \
#   python -u -m chocolatechip.model_training.train ...

python -u -m chocolatechip.model_training.train \
  --profile {experiment.profile} \
  --template {experiment.template} \
  --val-frac {experiment.val_frac} \
  --num-gpus {experiment.num_gpus} \
  --color-preset {experiment.color_preset} \
  --ultra-model {experiment.ultra_model} \
  --no-sweep
