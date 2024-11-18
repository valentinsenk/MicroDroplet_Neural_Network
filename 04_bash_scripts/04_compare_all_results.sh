#!/bin/bash

#SBATCH --nodes=1  # Run all processes on a single node
#SBATCH --nodelist=node04.imws.tuwien.ac.at #use the new node
#SBATCH --ntasks=2  # Use 2 cpu cores
#SBATCH --job-name=COMPARISON # Job Name
#SBATCH --time=0-02:00:00  # Time limit hrs:min:sec
#SBATCH --output=slurm-%j--%x.log  # Standard output and error log

####### variables #######
comparisonfile='compare_results_BASH_v1'
matlab_script_dir='/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/02_scripts/03_postprocessing/'
root_dir='/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/01_data/parameter_files/geometrical_samples/v8-3'
##########################

# Print running host
hostname

# Print postprocessing
echo "Start time POSTPROCESSING: $(date -Isec)"
start_post=$(date +%s)
echo "##########################"

# COMPARE WITH MATLAB
/opt/MATLAB/R2023b/bin/matlab -nosplash > matlab_${SLURM_JOB_NAME}.log << EOF
addpath('$matlab_script_dir')
root_dir='$root_dir'
$comparisonfile
exit
EOF

echo "##########################"
echo "Stop time: $(date -Isec)" 

stop_post=$(date +%s)
elapsed_post=$(echo "scale=2; ($stop_post - $start_post) / 3600" | bc)
echo "Elapsed time for POSTPROCESSING: $elapsed_post hours"

