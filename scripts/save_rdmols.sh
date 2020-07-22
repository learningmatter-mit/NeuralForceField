#!/bin/bash
#SBATCH -p sched_mit_rafagb,sched_mit_rafagb_amd
#SBATCH -t 4300
#SBATCH --mem-per-cpu 5G
#SBATCH -n 10

source $HOME/.bashrc
export NFFDIR=/home/saxelrod/repo/nff/covid/NeuralForceField
export PYTHONPATH="$NFFDIR:$PYTHONPATH"

source activate htvs
python save_rdmols.py
