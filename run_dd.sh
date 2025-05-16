for nu in 100 0.5 1.5 2.5 10
do
  for data in super temp syn power compactiv
  do
    sbatch --array=0-99 start_dd.sh data=\"$data\" nu=$nu
  done
done
