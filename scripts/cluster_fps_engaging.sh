#!/bin/bash
#SBATCH -p sched_mit_rafagb,sched_mit_rafagb_amd,sched_opportunist
#SBATCH -t 10080
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --mem-per-cpu 5G

source $HOME/.bashrc
export PYTHONPATH="/home/saxelrod/Repo/projects/covid_nff/NeuralForceField:${PYTHONPATH}"

source deactivate
source activate /home/saxelrod/miniconda3/envs/htvs

python cluster_fps.py --arg_path cluster_fps_engaging.json
