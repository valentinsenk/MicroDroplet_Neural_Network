#!/bin/bash
#SBATCH --nodes=1  # Run all processes on a single node
#SBATCH --nodelist=node04.imws.tuwien.ac.at #use the new node
#SBATCH --ntasks=2  # Use 2 threads (=1 cpu core)
#SBATCH --array=1-100%20 #submit job array with a maximum of 20 parallel jobs
#SBATCH --job-name=build_abq_model # Job Name
#SBATCH --time=01:00:00
#SBATCH --output=/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/01_data/parameter_files/geometrical_samples/v1/%03a/slurm-%A_%a--%x.log  # Standard output and error log
##SBATCH --licenses="abaqus_teaching@2501@l4.zserv.tuwien.ac.at":6

### current master root where you are in should be:
# /home/vsenk/Droplet_Tests_FEA/01_neural_network_project/

# Define the root directory where the parameter directories are located
ROOT_DIR="/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/01_data/parameter_files/geometrical_samples/v1"
# Get the current parameter directory based on the job array index
PARAM_DIR=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
# Navigate to the directory
cd $ROOT_DIR/$PARAM_DIR

# Print running host
hostname
# Print start time
echo "Start time BUILD ABQ MODEL: $(date -Isec)"
start_model=$(date +%s)


# if abaqus_teaching license server is not yet in abaqus_v6.env then add it
touch abaqus_v6.env
if ! grep -q "abaquslm_license_file" abaqus_v6.env ; then
	echo 'abaquslm_license_file="2501@l4.zserv.tuwien.ac.at"' >> abaqus_v6.env # can be 2501@lic-srv1.it.tuwien.ac.at (research) or 2501@l4.zserv.tuwien.ac.at (teaching)
fi
if ! grep -q "academic" abaqus_v6.env ; then
	echo 'academic=TEACHING' >> abaqus_v6.env # can be RESEARCH or TEACHING
fi

# Run the Abaqus Python script to build the model
/opt/abaqus/Commands/abq2024 cae noGUI=/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/02_scripts/02_preprocessing/01_build_model_v1.py

# Wait for the Abaqus job to finish
wait

# Run the Julia script to clean and process the inp files
/home/vsenk/bin/julia-1.8.1/bin/julia /home/vsenk/Droplet_Tests_FEA/01_neural_network_project/02_scripts/02_preprocessing/02_build_clean_inp_files_v1.jl

wait

# Record end time and calculate elapsed time
echo "Stop time BUILD ABQ MODEL: $(date -Isec)"
stop_model=$(date +%s)
elapsed_model=$(echo "scale=2; ($stop_model - $start_model) / 3600" | bc)

# Print elapsed time
echo "Elapsed time BUILD ABQ MODEL: $elapsed_model hours"