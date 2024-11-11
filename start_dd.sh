#!/usr/bin/env bash
#SBATCH -A #####
#SBATCH -t 02:00:00
#SBATCH -o out_dd.txt
#SBATCH -e err_dd.txt
#SBATCH -n 1

ml python/3.12.3
source ~/my_python/bin/activate
python dd.py seed=$SLURM_ARRAY_TASK_ID $*
