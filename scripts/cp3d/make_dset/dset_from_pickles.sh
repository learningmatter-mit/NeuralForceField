#!/bin/bash
#SBATCH -p sched_mit_rafagb
#SBATCH -t 4300
#SBATCH -n 64
#SBATCH -N 1
#SBATCH --mem 350G

source deactivate
source ~/.bashrc
source activate nff
export PYTHONPATH="$HOME/Repo/projects/covid_clean/NeuralForceField:${PYTHONPATH}"

# `jq` allows you to read a JSON file in bash. Here we are using it to get the number of threads from the config file. If you don't have `jq` installed,
# you can just change `num_threads` manually here, or you can download it by running `bash download_jq.sh`

NUM_THREADS=$(cat dset_config.json | $jq ".num_threads")
END=$((NUM_THREADS-1))

for i in $(seq 0 $END);
do
cmd="python dset_from_pickles.py --thread $i --config_file dset_config.json"
echo $cmd
eval $cmd
done
