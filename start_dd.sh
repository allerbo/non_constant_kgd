#!/usr/bin/env bash
#SBATCH -A ####
#SBATCH -t 00:40:00
#SBATCH -o out_dd.txt
#SBATCH -e err_dd.txt
#SBATCH -n 1


ml SciPy-bundle/2022.05-foss-2022a
source ~/my_python/bin/activate
python double_descent.py $*
