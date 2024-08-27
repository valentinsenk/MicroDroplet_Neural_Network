#!/bin/bash
#SBATCH --nodes=1  # Run all processes on a single node
#SBATCH --nodelist=node04.imws.tuwien.ac.at #use the new node
#SBATCH --ntasks=2  # Use 2 threads (=1 cpu core)
#SBATCH --array=1-100%20 #submit job array with a maximum of 20 parallel jobs
#SBATCH --job-name=postprocess_abq_results # Job Name
#SBATCH --time=01:00:00
##SBATCH --output=/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/01_data/parameter_files/geometrical_samples/v1/%03a/slurm-%A_%a--%x.log  # Standard output and error log
#SBATCH --output=/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/01_data/parameter_files/mechanical_samples/v1/%03a/slurm-%A_%a--%x.log  # Standard output and error log
#SBATCH --licenses="abaqus_teaching@2501@l4.zserv.tuwien.ac.at":6

### current master root where you are in should be:
# /home/vsenk/Droplet_Tests_FEA/01_neural_network_project/

# Define the root directory where the parameter directories are located
#ROOT_DIR="/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/01_data/parameter_files/geometrical_samples/v1"
ROOT_DIR="/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/01_data/parameter_files/mechanical_samples/v1"
# Get the current parameter directory based on the job array index
PARAM_DIR=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
# Navigate to the directory
cd $ROOT_DIR/$PARAM_DIR

# Get the output file name based on the job array index and version
output_file="lhs_${PARAM_DIR}_v1"

# Print running host
hostname
# Print start time
echo "Start POSTPROCESSING: $(date -Isec)"
start_postpr=$(date +%s)

# GET RESULTS VIA ABQ-PYTHON-SCRIPT
/opt/abaqus/Commands/abq2024 cae noGUI=/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/02_scripts/03_postprocessing/00_get_results_from_abq_v1.py -- "$output_file"

wait

# MAKE SOME FURTHER POSTPROCESSING AND PLOTS WITH MATLAB
matlab_stresses='process_energy_data_v1'
matlab_energies='porcess_force_disp_data_v1'

/opt/MATLAB/R2023b/bin/matlab -nosplash > matlab_${SLURM_JOB_NAME}.log << EOF
addpath('/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/02_scripts/03_postprocessing/')
jobname=$output_file
inc=10
$matlab_stresses
$matlab_energies
exit
EOF

# Record end time and calculate elapsed time
echo "Stop time BUILD ABQ MODEL: $(date -Isec)"
stop_postpr=$(date +%s)
elapsed_time=$(echo "scale=2; ($stop_postpr - $start_postpr) / 3600" | bc)

# Print elapsed time
echo "Elapsed time BUILD ABQ MODEL: $elapsed_time hours"