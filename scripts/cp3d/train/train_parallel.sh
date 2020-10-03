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

# change to the number of GPUs you're using per node
export SLURM_GPUS_PER_NODE=2
export LD_LIBRARY_PATH=lib/$CONDA_PREFIX/:$LD_LIBRARY_PATH

# change to your location of NeuralForceField
export NFFDIR="$HOME/repo/nff/covid_clean/NeuralForceField"
export PYTHONPATH=$NFFDIR:$PYTHON_PATH

python $NFFDIR/scripts/cp3d/train/train_parallel.py train_config.json  & pid=$!
wait


