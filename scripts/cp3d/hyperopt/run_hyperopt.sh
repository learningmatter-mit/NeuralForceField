#!/bin/bash
#SBATCH -N 1
#SBATCH -t 10000
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH -p sched_mit_rafagb_amd,sched_mit_rafagb

source deactivate
source ~/.bashrc

CONFIG="config/cp3d_single_cov2_gen.json"

# change to the number of GPUs you're using per node
export SLURM_GPUS_PER_NODE=1
export LD_LIBRARY_PATH=lib/$CONDA_PREFIX/:$LD_LIBRARY_PATH

# change to your location of NeuralForceField
export NFFDIR="$HOME/repo/nff/master/NeuralForceField"
export PYTHONPATH=$NFFDIR:$PYTHON_PATH

source activate nff

cmd="python run_hyperopt.py --config_file $CONFIG"
echo $cmd
eval $cmd
