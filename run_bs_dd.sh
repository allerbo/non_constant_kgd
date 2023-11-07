for day in {1..366}
do
  for nu in 100 10 2.5 1.5 0.5
  do
    sbatch start_dd.sh seed=$day nu=$nu data=\"bs\"
  done
done
