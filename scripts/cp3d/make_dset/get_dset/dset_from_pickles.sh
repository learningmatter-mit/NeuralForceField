#!/bin/bash
#SBATCH -N 1
#SBATCH -t 30240
#SBATCH --mem=300G
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --ntasks-per-node 40
#SBATCH --qos=high
#SBATCH -p normal
#SBATCH --constraint=xeon-g6

source deactivate
source ~/.bashrc
source activate nff

# change to your nff directory
export NFFDIR="$HOME/repo/nff/covid_clean/NeuralForceField"
export PYTHONPATH="$NFFDIR:$PYTHONPATH"

# `jq` allows you to read a JSON file in bash. Here we are using it to get the number of threads from the config file. If you don't have `jq` installed,
# you can just change `num_threads` manually here, or you can download it by running `cd ../.. && bash download_jq.sh && cd - && source ~/.bashrc `

NUM_THREADS=$(cat dset_config.json | $jq ".num_threads")
END=$((NUM_THREADS-1))

# increase the resource limit to avoid "too many open files" errors during parallel feature generation
ulimit -n 50000

for i in $(seq 0 $END);
do
cmd="python dset_from_pickles.py --thread $i --config_file dset_config.json"
echo $cmd
eval $cmd
done
