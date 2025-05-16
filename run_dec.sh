for nu in 100 0.5 1.5 2.5 10
do
  for data in temp super power
  do
    sbatch --array=0-199 start_dec.sh data=\"$data\" nu=$nu
  done
  sbatch --array=0-99 start_dec.sh data=\"compactiv\" nu=$nu
done
