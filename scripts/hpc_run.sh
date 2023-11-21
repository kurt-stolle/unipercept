#!/bin/bash
# Script for training a model in a HPC environment with Slurm and multiple GPUs.
# Tested on the Snellius cluster provided by SURF.

#SBATCH --mail-type=ALL
#SBATCH --partition=gpu --nodes 1

# Exit immediately if a command exits with a non-zero status.
set -e

# Environment setup
SCRIPTS_DIR=$(dirname "$0")
source "${SCRIPTS_DIR}/hpc_env.sh"

# Get CLI command path
CLI_PATH=$(which unicli)

# Run
$PYTHON -m accelerate.commands.launch "${CLI_PATH}" $@

# Exit
exit $?
