#!/bin/bash
#SBATCH -p sched_mit_rafagb
#SBATCH -t 4300
#SBATCH -n 64
#SBATCH -N 1
#SBATCH --mem 350G

source deactivate
source activate nff


cmd="python random_search.py --config_file search_config.json"
echo $cmd
eval $cmd