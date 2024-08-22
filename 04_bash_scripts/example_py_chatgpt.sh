#!/bin/bash
#SBATCH --job-name=model_generation
#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --output=logs/model_generation_%A_%a.out

# Load the necessary modules (if any)
module load abaqus

# Define the root directory where the parameter directories are located
ROOT_DIR="/path/to/your/project_root/01_data/parameter_files/geometrical_samples/v1/"

# Get the current parameter directory based on the job array index
PARAM_DIR=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

# Navigate to the directory
cd $ROOT_DIR/$PARAM_DIR

# Run the Python script using Abaqus Python
abaqus python /path/to/your/project_root/02_scripts/01_build_model_v1.py
