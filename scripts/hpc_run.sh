#!/bin/bash
# Script for training a model in a HPC environment with Slurm and multiple GPUs.
# Tested on the Snellius cluster provided by SURF.

#SBATCH --mail-type=ALL
#SBATCH --partition=gpu --nodes 1

set -e

echo "Running on $(hostname)"

echo "Loading HPC modules"
source "./scripts/hpc_env.sh"

echo "Loading Python virtual environment"
source "./venv/bin/activate"

echo "Starting distributed training"
accelerate launch $(which unicli) $@
exit $?
