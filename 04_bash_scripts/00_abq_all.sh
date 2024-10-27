#!/bin/bash
#SBATCH --nodes=1  # Run all processes on a single node
#SBATCH --nodelist=node04.imws.tuwien.ac.at #use the new node
#SBATCH --ntasks=2  # Use 4 threads (=2 cpu core)
#SBATCH --array=1-150%32 #submit job array with a maximum of 20 parallel jobs
#SBATCH --job-name=selected2_v1 # Job Name
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/01_data/parameter_files/selected_param_samples2/v2/%03a/slurm-%A_%a--%x.log  # Standard output and error log
#SBATCH --licenses="abaqus_teaching@2501@l4.zserv.tuwien.ac.at":5   ##SBATCH --licenses="abaqus@2501@lic-srv1.it.tuwien.ac.at":5


# Define the root directory where the parameter directories are located
ROOT_DIR="/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/01_data/parameter_files/selected_param_samples2/v2"
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
elapsed_time=$(echo "scale=2; ($stop_model - $start_model) / 3600" | bc)

# Print elapsed time
echo "Elapsed time BUILD ABQ MODEL: $elapsed_time hours"

#########################################################################

# Get the input file name based on the job array index and version
input_file="lhs_${PARAM_DIR}_v1.inp"

# Print start time
echo "Start time RUN ABQ: $(date -Isec)"
start_abq=$(date +%s)

# Run abaqus solver
unbuffer /opt/abaqus/Commands/abq2024 scratch="/tmp" job="lhs_${PARAM_DIR}_v1" cpus=1 mp_mode=threads input=$input_file standard_parallel=solver interactive

wait

# Record end time and calculate elapsed time
echo "Stop time RUN ABQ: $(date -Isec)"
stop_abq=$(date +%s)
elapsed_time=$(echo "scale=2; ($stop_abq - $start_abq) / 3600" | bc)

# Print elapsed time
echo "Elapsed time BUILD ABQ MODEL: $elapsed_time hours"

########################################################################

# Get the output file name based on the job array index and version
output_file="lhs_${PARAM_DIR}_v1"

# Print start time
echo "Start time POSTPROCESSING: $(date -Isec)"
start_postpr=$(date +%s)

# GET RESULTS VIA ABQ-PYTHON-SCRIPT
/opt/abaqus/Commands/abq2024 cae noGUI=/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/02_scripts/03_postprocessing/00_get_results_from_abq_v1.py -- "$output_file"

wait

# MAKE SOME FURTHER POSTPROCESSING AND PLOTS WITH MATLAB
matlab_stresses='process_energy_data_v1'
matlab_energies='process_force_disp_data_v1'

/opt/MATLAB/R2023b/bin/matlab -nosplash > matlab_${SLURM_JOB_NAME}.log << EOF
addpath('/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/02_scripts/03_postprocessing/')
jobname='$output_file'
inc=10
$matlab_stresses
$matlab_energies
exit
EOF

wait

# Record end time and calculate elapsed time
echo "Stop time POSTPROCESSING: $(date -Isec)"
stop_postpr=$(date +%s)
elapsed_time=$(echo "scale=2; ($stop_postpr - $start_postpr) / 3600" | bc)

# Print elapsed time
echo "Elapsed time BUILD ABQ MODEL: $elapsed_time hours"

#######################################################################

echo "Model lhs_${PARAM_DIR}_v1 finished."

stop_all=$(date +%s)
elapsed_time=$(echo "scale=2; ($stop_postpr - $start_model) / 3600" | bc)

# Print elapsed time ALL
echo "Elapsed time BUILD + ABQ SOLVER + POSTPROCESS: $elapsed_time hours"