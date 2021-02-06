#!/bin/bash
#SBATCH -N 1
#SBATCH -t 10000
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH -p sched_mit_rafagb_amd,sched_mit_rafagb


source deactivate
source ~/.bashrc
source activate nff

# CONFIG="../train_config/ schnet_holdout_zhu.json"
CONFIG="../train_config/ schnet_rand_zhu.json"

# change to the number of GPUs you're using per node
export SLURM_GPUS_PER_NODE=1
export LD_LIBRARY_PATH=lib/$CONDA_PREFIX/:$LD_LIBRARY_PATH

# change to your location of NeuralForceField
export NFFDIR="$HOME/repo/nff/master/NeuralForceField"
export PYTHONPATH=$NFFDIR:$PYTHON_PATH

cmd="python train_parallel.py $CONFIG  "
echo $cmd
eval $cmd


