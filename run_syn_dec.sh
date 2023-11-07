for seed in {0..99}
do
  for nu in 100 10 2.5 1.5 0.5
  do
    for data in \"lin_sin\" \"two_freq\"
    do
      sbatch start_syn_dec.sh seed=$seed nu=$nu data=$data
    done
  done
done
