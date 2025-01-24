#!/bin/bash

#SBATCH --nodes=1
#SBATCH --nodelist=node04.imws.tuwien.ac.at
#SBATCH --ntasks=2
#SBATCH --job-name=CREATE_COLLAGE
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm-%j--%x.log

####### variables #######
python_script_dir='/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/02_scripts/03_postprocessing'
root_dir='/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/01_data/parameter_files/new_mechanical_samples/v1'
output_dir="${root_dir}/_compare_allresults"
output_collage="${output_dir}/collage.png"
##########################

# Print running host
hostname

# Print postprocessing start time
echo "Start time POSTPROCESSING: $(date -Isec)"
start_post=$(date +%s)
echo "##########################"

# Create the _compare_results directory if it doesn't exist
mkdir -p $output_dir

# Activate the Conda environment
source ~/miniconda3/bin/activate myenv

# RUN PYTHON SCRIPT TO CREATE COLLAGE
python3 $python_script_dir/create_collage.py --root_dir $root_dir --output $output_collage

echo "##########################"
echo "Stop time: $(date -Isec)"

stop_post=$(date +%s)
elapsed_post=$(echo "scale=2; ($stop_post - $start_post) / 3600" | bc)
echo "Elapsed time for POSTPROCESSING: $elapsed_post hours"
