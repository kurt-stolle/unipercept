#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=18
#SBATCH --time=24:00:00
#SBATCH --job-name=unipercept


set -e
echo "Running on $(hostname)"
echo "Loading HPC modules"
source "./scripts/hpc_env.sh"
echo "Starting distributed training"

# ./venv/bin/accelerate launch `realpath ./venv/bin/unicli` $@
srun `realpath ./venv/bin/accelerate` launch `realpath ./venv/bin/unicli` $@
exit $?
