#!/bin/bash
#SBATCH -N 1
#SBATCH -t 30240
##SBATCH --gres=gpu:2
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

export PYTHONPATH="$HOME/Repo/projects/covid_clean/NeuralForceField:${PYTHONPATH}"

python trim_confs.py --from_model_path $HOME/rgb_nfs/models/cov_1_combined --to_model_path $HOME/rgb_nfs/models/cov_1_single_conf --num_confs 1
