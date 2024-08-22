#!/bin/bash

#SBATCH --nodes=1  # Run all processes on a single node
#SBATCH --nodelist=node04.imws.tuwien.ac.at #use the new node
#SBATCH --ntasks=4  # Use 4 cpu cores
#SBATCH --job-name=matlab_postpr # Job Name
#SBATCH --qos=normal  # "normal" or "important". Use "important" to jump the normal "queue" for important small and short jobs.
#SBATCH --time=3-00:00:00  # Time limit hrs:min:sec
#SBATCH --output=slurm-%j--%x.log  # Standard output and error log
##SBATCH --licenses="abaqus@2501@lic-srv1.it.tuwien.ac.at":12 # Request 8 abaqus licenses


####### variables #######
inc=10 # for plots in matlab
matlab_script='force_disp.m'
##########################

# Loop over all directories with "_results" suffix
for dir in *_results; do
  # Extract the job name by removing the "_results" suffix
  jobname="${dir%_results}"

  # Run MATLAB script for each directory
  /opt/MATLAB/R2023b/bin/matlab -nosplash -r "addpath('00_scripts'); jobname='${jobname}'; inc=${inc}; run('${matlab_script}'); exit;" > matlab_${jobname}_results.log

done