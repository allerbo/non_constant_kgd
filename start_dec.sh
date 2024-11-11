#!/usr/bin/env bash
#SBATCH -A #####
#SBATCH -t 1:00:00
#SBATCH -o out_dec.txt
#SBATCH -e err_dec.txt
#SBATCH -n 1


ml python/3.12.3
source ~/my_python/bin/activate
python dec.py seed=$SLURM_ARRAY_TASK_ID $*
