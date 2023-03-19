#!/bin/bash
# ====================
# Options for sbatch
# ====================

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=4000

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=2

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=04:00:00

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
CONDA_ENV_NAME=ip
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}


# =================================
# Move input data to scratch disk
# =================================
# Move data from a source location, probably on the distributed filesystem
# (DFS), to the scratch space on the selected node. Your code should read and
# write data on the scratch space attached directly to the compute node (i.e.
# not distributed), *not* the DFS. Writing/reading from the DFS is extremely
# slow because the data must stay consistent on *all* nodes. This constraint
# results in much network traffic and waiting time for you!
#
# This example assumes you have a folder containing all your input data on the
# DFS, and it copies all that data  file to the scratch space, and unzips it. 
#
# For more guidelines about moving files between the distributed filesystem and
# the scratch space on the nodes, see:
#     http://computing.help.inf.ed.ac.uk/cluster-tips

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS - change line below if loc different
src_path1=/home/${USER}/ipdis/data/extra_data/MRI_IP_project/WMH_challenge_dataset/public/GE3T
src_path2=/home/${USER}/ipdis/data/extra_data/MRI_IP_project/WMH_challenge_dataset/public/Singapore
src_path3=/home/${USER}/ipdis/data/extra_data/MRI_IP_project/WMH_challenge_dataset/public/Utrecht

# input data directory path on the scratch disk of the node
dest_path=${SCRATCH_HOME}/ipdis/preprep/in_data
mkdir -p ${dest_path}  # make it if required

# Important notes about rsync:
# * the --compress option is going to compress the data before transfer to send
#   as a stream. THIS IS IMPORTANT - transferring many files is very very slow
# * the final slash at the end of ${src_path}/ is important if you want to send
#   its contents, rather than the directory itself. For example, without a
#   final slash here, we would create an extra directory at the destination:
#       ${SCRATCH_HOME}/project_name/data/input/input

WMH=WMH_challenge_dataset
mkdir -p ${dest_path}/${WMH}/public/GE3T
mkdir -p ${dest_path}/${WMH}/public/Singapore
mkdir -p ${dest_path}/${WMH}/public/Utrecht
rsync --archive --update --compress --progress ${src_path1}/ ${dest_path}/${WMH}/public/GE3T
rsync --archive --update --compress --progress ${src_path2}/ ${dest_path}/${WMH}/public/Singapore
rsync --archive --update --compress --progress ${src_path3}/ ${dest_path}/${WMH}/public/Utrecht

# make directory for preprocessing results
result_path=${SCRATCH_HOME}/ipdis/preprep/out_data/${WMH}
mkdir -p ${result_path}

# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

experiment_text_file=$1
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
# script_dir=/home/${USER}/ipdis/Trustworthai-MRI-WMH/twaidata/MRI_preprep
# COMMAND="python ${script_dir}/simple_preprocess.py -i ${dest_path} -o ${result_path} -n ${WMH} -s '0' -e '-1'"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"


# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"

src_path=${result_path}
copyback_path=/home/${USER}/ipdis/data/preprocessed_data
rsync --archive --update --compress --progress ${src_path}/ ${copyback_path}

# ================================
# Delete data from scratch space
# ================================
# echo "Deleting data on scratch space"

# commented out so that multiple scripts can share the same data without
# crashing each other. will need to delete data manually afterwards tho.
# delete input data
# rm -rf ${result_path}

# #delete output results
# rm -rf ${dest_path}

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
