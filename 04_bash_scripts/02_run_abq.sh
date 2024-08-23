#!/bin/bash
#SBATCH --nodes=1  # Run all processes on a single node
#SBATCH --nodelist=node04.imws.tuwien.ac.at #use the new node
#SBATCH --ntasks=4  # Use 4 threads (=2 cpu core)
#SBATCH --array=1-100%20 #submit job array with a maximum of 20 parallel jobs
#SBATCH --job-name=run_geom_samples # Job Name
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/01_data/parameter_files/geometrical_samples/v1/%03a/slurm-%A_%a--%x.log  # Standard output and error log
#SBATCH --licenses="abaqus_teaching@2501@l4.zserv.tuwien.ac.at":6

# Define the root directory where the parameter directories are located
ROOT_DIR="/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/01_data/parameter_files/geometrical_samples/v1"
# Get the current parameter directory based on the job array index
PARAM_DIR=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
# Navigate to the directory
cd $ROOT_DIR/$PARAM_DIR

# Get the input file name based on the job array index and version
input_file="lhs_${PARAM_DIR}_v1.inp"

# Print running host
hostname
# Print start time
echo "Start time RUN ABQ: $(date -Isec)"
start_abq=$(date +%s)

# Run abaqus solver
unbuffer /opt/abaqus/Commands/abq2024 scratch="/tmp" job="lhs_${PARAM_DIR}_v1" cpus=2 mp_mode=threads input=$input_file standard_parallel=solver interactive

wait

# Record end time and calculate elapsed time
echo "Stop time BUILD ABQ MODEL: $(date -Isec)"
stop_abq=$(date +%s)
elapsed_time=$(echo "scale=2; ($stop_abq - $start_abq) / 3600" | bc)

# Print elapsed time
echo "Elapsed time BUILD ABQ MODEL: $elapsed_time hours"