#!/bin/bash
#SBATCH -N 1
#SBATCH -t 10000
#SBATCH --mem=300G
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --mem-per-cpu=5G
#SBATCH -p sched_mit_rafagb_amd,sched_mit_rafagb


source deactivate
source ~/.bashrc
source activate htvs

# change as necessary
export HTVSDIR="/home/saxelrod/repo/htvs/master/htvs"
export DJANGOCHEMDIR="/home/saxelrod/repo/htvs/master/htvs/djangochem"
export NFFDIR="/home/saxelrod/repo/nff/master/NeuralForceField"

python make_dset.py
