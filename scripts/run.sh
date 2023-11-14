#!/bin/bash
# Script for training a model in a HPC environment with Slurm and multiple GPUs.
# Tested on the Snellius cluster provided by SURF.

#SBATCH --mail-type=ALL
#SBATCH --partition=gpu --nodes 1

# Exit immediately if a command exits with a non-zero status.
set -e

# Check whether the environment variable `UNI_COMMAND` is set, this should be provided in the `sbatch` invocation using
# the `--export=UNI_COMMAND=...` flag.
if [ -z "$UNI_COMMAND" ]; then
    echo "Environment variable UNI_COMMAND is not set, please provide a path to a config file."
    exit 1
fi
if [ ! -f "$UNI_ENVIRONMENT" ]; then
    UNI_ENVIRONMENT="unipercept"
    echo "Using default conda environment: ${UNI_ENVIRONMENT}"
fi

echo "Starting UniPercept with command: \`unicli ${UNI_COMMAND}\`"

# Modules provided by the cluster.
module purge
module load 2022
module load Miniconda3/4.12.0

# Run the training script using Accelerate
# UNI_SCRIPTS="$( dirname "$( realpath "${BASH_SOURCE[0]}" )" )"
UNI_CLI="$(conda run -n "${UNI_ENVIRONMENT}" which unicli)"
conda run -n "${UNI_ENVIRONMENT}" accelerate launch "${UNI_CLI}" ${UNI_COMMAND}

# Exit
exit $?
