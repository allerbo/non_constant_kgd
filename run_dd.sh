for nu in 0.5 1.5 2.5 10 100
do
  for data in super temp syn power compactiv wood
  do
    sbatch --array=0-99 start_dd.sh data=\"$data\" nu=$nu
  done
done
