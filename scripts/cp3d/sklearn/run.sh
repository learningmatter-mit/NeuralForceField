#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 4300
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 15
#SBATCH -p sched_mit_rafagb_amd,sched_mit_rafagb

source deactivate
source activate nff

export PYTHONPATH="/home/saxelrod/Repo/projects/master/NeuralForceField:$PYTHONPATH"
CONFIG="config/kernel_ridge/confs_mae.json"

cmd="python run.py --config_file $CONFIG"
echo $cmd
eval $cmd
