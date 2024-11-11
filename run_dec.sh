for nu in 0.5 1.5 2.5 10 100
do
  for data in wood temp super power
  do
    sbatch --array=0-199 start_dec.sh data=\"$data\" nu=$nu
  done
  sbatch --array=0-99 start_dec.sh data=\"compactiv\" nu=$nu
done
