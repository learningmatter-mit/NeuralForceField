#!/bin/bash
#SBATCH -N 4
#SBATCH -t 10000
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH -p sched_mit_rafagb_amd,sched_mit_rafagb

source deactivate
source ~/.bashrc
source activate nff

CONFIG="config/cp3d_ndu_cov2_gen.json"
# CONFIG="config/cp3d_single_cov2_gen.json"

# change to the number of GPUs you're using per node
export SLURM_GPUS_PER_NODE=1
export LD_LIBRARY_PATH=lib/$CONDA_PREFIX/:$LD_LIBRARY_PATH

# change to your location of NeuralForceField
export NFFDIR="$HOME/repo/nff/master/NeuralForceField"
export PYTHONPATH=$NFFDIR:$PYTHON_PATH

cmd="python $NFFDIR/scripts/cp3d/train/train_parallel.py $CONFIG  " # & pid=\$!"
echo $cmd
eval $cmd
# wait


