#!/bin/bash

#SBATCH --nodes=1  # Run all processes on a single node
#SBATCH --nodelist=node04.imws.tuwien.ac.at #use the new node
#SBATCH --ntasks=2  # Use 4 cpu cores
#SBATCH --job-name=COMPARISON # Job Name
#SBATCH --qos=normal  # "normal" or "important". Use "important" to jump the normal "queue" for important small and short jobs.
#SBATCH --time=3-00:00:00  # Time limit hrs:min:sec
#SBATCH --output=slurm-%j--%x.log  # Standard output and error log
##SBATCH --licenses="abaqus@2501@lic-srv1.it.tuwien.ac.at":12 # Request 8 abaqus licenses

####### variables #######
comparisonfile='compare_results_NEW2'
##########################

# Print running host
hostname

# Print postprocessing
echo "Start time POSTPROCESSING: $(date -Isec)"
start_post=$(date +%s)
echo "##########################"

# COMPARE WITH MATLAB
#
/opt/MATLAB/R2023b/bin/matlab -nosplash > matlab_${SLURM_JOB_NAME}.log << EOF
addpath('00_scripts')
$comparisonfile
exit
EOF

echo "##########################"
echo "Stop time: $(date -Isec)" 

stop_post=$(date +%s)
elapsed_post=$(echo "scale=2; ($stop_post - $start_post) / 3600" | bc)
echo "Elapsed time for POSTPROCESSING: $elapsed_post hours"

overall_elapsed=$(echo "scale=2; ($stop_post - $start_abq) / 3600" | bc)
echo "Overall elapsed time: $overall_elapsed hours"

