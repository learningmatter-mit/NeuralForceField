#!/bin/bash
#SBATCH -N 8
#SBATCH -t 30240
#SBATCH --gres=gpu:2
#SBATCH --mem=300G
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 20
#SBATCH --gres=gpu:volta:2
#SBATCH --qos=high
#SBATCH -p normal
#SBATCH --constraint=xeon-g6

source deactivate
source ~/.bashrc
source activate nff

export MASTER_ADDR=$(srun -l bash -c 'hostname' | sort | head -1 | awk '{print $2}')
export MASTER_IP=$(getent hosts `hostname` | cut -d ' ' -f1)
export MASTER_PORT=8888
# change to the number of GPUs you're using per node
export SLURM_GPUS_PER_NODE=2
export LD_LIBRARY_PATH=lib/$CONDA_PREFIX/:$LD_LIBRARY_PATH
export NFFDIR="$HOME/repo/nff/covid_clean/NeuralForceField"

# this is the problem: something's going wrong in this branch of covid_clean
# export PYTHONPATH=$NFFDIR:$PYTHON_PATH
export PYTHONPATH="$PYTHONPATH:$DJANGOCHEMDIR:$HTVSDIR"

python $NFFDIR/scripts/cp3d/train/train_parallel.py job_info.json  & pid=$!
wait


