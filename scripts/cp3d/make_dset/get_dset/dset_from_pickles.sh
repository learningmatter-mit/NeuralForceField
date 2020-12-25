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

CONFIG="config/cov_2_3cl_test.json"

# change to your nff directory
export NFFDIR="$HOME/Repo/projects/master/NeuralForceField"
export PYTHONPATH="$NFFDIR:$PYTHONPATH"

NUM_THREADS=$(cat $CONFIG | jq ".num_threads")
SLURM_PAR=$(cat $CONFIG | jq ".slurm_parallel")
END=$((NUM_THREADS-1))

# increase the resource limit to avoid "too many open files" errors during parallel feature generation
ulimit -n 50000

for i in $(seq 0 $END); do
    cmd='python dset_from_pickles.py --thread '$i' --config_file '$CONFIG' '

    # if parallelizing over nodes with slurm, use srun and let everything run together
    if [ $SLURM_PAR = 'true' ]; then
        cmd="srun -N 1 "$cmd

        # only produce the output from threads not run at the same time
        if (( $i % $SLURM_NNODES != 0)); then
            cmd=$cmd" > /dev/null 2>&1"
        fi
        j=$(( i + 1 ))
        cmd=$cmd" & pids[${j}]=\$!"

    fi

    echo $cmd
    eval $cmd

done

# wait for all pids
for pid in ${pids[*]}; do
    cmd="wait $pid"
    echo $cmd
    eval $cmd
done
