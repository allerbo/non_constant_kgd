for seed in {0..99}
do
  for nu in 100 10 2.5 1.5 0.5
  do
    sbatch start_dd.sh seed=$seed nu=$nu data=\"syn\"
  done
done
