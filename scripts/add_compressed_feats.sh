#!/bin/bash
#SBATCH -p sched_mit_rafagb
#SBATCH -t 4300
#SBATCH -n 64
#SBATCH -N 1
#SBATCH --mem 350G

source deactivate
source activate covid_mit
python add_compressed_feats.py --save_singles

# Note - we won't be able to store all of this as a single NFF dataset.
# We'll have to store multiple separate datasets that will be loaded
# by each of the separate nodes during training. 