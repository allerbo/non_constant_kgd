#!/usr/bin/env bash
#SBATCH -A ####
#SBATCH -t 0:20:00
#SBATCH -o out_syn_dec.txt
#SBATCH -e err_syn_dec.txt
#SBATCH -n 1


ml SciPy-bundle/2022.05-foss-2022a
source ~/my_python/bin/activate
python syn_dec.py $*
