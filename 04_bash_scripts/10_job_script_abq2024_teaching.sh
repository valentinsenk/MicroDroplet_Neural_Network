#!/bin/bash

#SBATCH --nodes=1  # Run all processes on a single node
#SBATCH --nodelist=node04.imws.tuwien.ac.at #use the new node
#SBATCH --ntasks=4  # Use 4 cpu cores
#SBATCH --job-name=Meniscus90_02rf_G0_03_v1 # Job Name
#SBATCH --qos=normal  # "normal" or "important". Use "important" to jump the normal "queue" for important small and short jobs.
#SBATCH --time=1-00:00:00  # Time limit hrs:min:sec
#SBATCH --output=slurm-%j--%x.log  # Standard output and error log
##SBATCH --licenses="abaqus@2501@lic-srv1.it.tuwien.ac.at":12 # Request 8 abaqus licenses
#SBATCH --licenses="abaqus_teaching@2501@l4.zserv.tuwien.ac.at":6 # Request 8 abaqus_teaching licenses (make sure abaqus.env is in folder)

####### variables #######
inc=10 #for plots in matlab
matlab_stresses='force_disp'
matlab_energies='energies'
##########################

# Print running host
hostname

# Print start time
echo "###--- JOB: $SLURM_JOB_NAME.inp ---###"
echo "##########################"

echo "Start time ABQ SOLVER: $(date -Isec)"
start_abq=$(date +%s)

# Run the abaqus command with usermaterial
unbuffer /opt/abaqus/Commands/abq2024 scratch="/tmp" job=$SLURM_JOB_NAME cpus=2 mp_mode=threads input=$SLURM_JOB_NAME.inp standard_parallel=solver interactive

wait

stop_abq=$(date +%s)
elapsed_abq=$(echo "scale=2; ($stop_abq - $start_abq) / 3600" | bc)

# Print time again
echo "##########################"
echo "Stop time ABQ SOLVER: $(date -Isec)"
echo "Elapsed time for ABQ SOLVER: $elapsed_abq hours"
echo "##########################"
# Print postprocessing
echo "Start time POSTPROCESSING: $(date -Isec)"
start_post=$(date +%s)
echo "##########################"

# GET RESULTS VIA ABQ-PYTHON-SCRIPT
/opt/abaqus/Commands/abq2024 cae noGUI=00_scripts/00_get_result_data_droplet_BC3.py -- "$SLURM_JOB_NAME"

# GET MATLAB GRAPHS
wait
#
/opt/MATLAB/R2023b/bin/matlab -nosplash > matlab_${SLURM_JOB_NAME}.log << EOF
addpath('00_scripts')
jobname='$SLURM_JOB_NAME'
inc=$inc
$matlab_stresses
$matlab_energies
exit
EOF

echo "##########################"
echo "Stop time: $(date -Isec)" 

stop_post=$(date +%s)
elapsed_post=$(echo "scale=2; ($stop_post - $start_post) / 3600" | bc)
echo "Elapsed time for POSTPROCESSING: $elapsed_post hours"

overall_elapsed=$(echo "scale=2; ($stop_post - $start_abq) / 3600" | bc)
echo "Overall elapsed time: $overall_elapsed hours"

