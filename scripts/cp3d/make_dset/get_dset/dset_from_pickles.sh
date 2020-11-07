#!/bin/bash
#SBATCH -N 8
#SBATCH -t 30240
#SBATCH --mem=300G
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --qos=high
#SBATCH -p normal
#SBATCH --constraint=xeon-g6

source deactivate
source ~/.bashrc
source activate nff

# change to your config file
CONFIG_FILE="files/cov_1.json"

# change to your nff directory
export NFFDIR="$HOME/repo/nff/covid_clean/NeuralForceField"
export PYTHONPATH="$NFFDIR:$PYTHONPATH"

# `jq` allows you to read a JSON file in bash. Here we are using it to get the number of threads from the config file. If you don't have `jq` installed,
# you can just change `num_threads` manually here, or you can download it by running `cd ../.. && bash download_jq.sh && cd - && source ~/.bashrc `

NUM_THREADS=$(cat $CONFIG_FILE | $jq ".num_threads")
SLURM_PAR=$(cat $CONFIG_FILE | $jq ".slurm_parallel")
END=$((NUM_THREADS-1))

# increase the resource limit to avoid "too many open files" errors during parallel feature generation
ulimit -n 50000

for i in $(seq 0 $END); do
    cmd="python dset_from_pickles.py --thread $i --config_file $CONFIG_FILE"

    # if parallelizing over nodes with slurm, use srun and let everything run together
    if [ $SLURM_PAR = 'true' ]; then
        cmd="srun -N 1 "$cmd

        # only produce the output from threads not run at the same time
        if (( $i % $SLURM_NNODES != 0)); then
            cmd=$cmd" > /dev/null 2>&1"
        fi

        cmd=$cmd" &"
    fi
    echo $cmd
    eval $cmd

done

wait
