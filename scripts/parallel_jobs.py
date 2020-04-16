import sys
sys.path.insert(0, "..")

import uuid
import os
import shutil

from nff.data.parallel import split_dataset
from nff.data.dataset import Dataset

TEMPLATE = """#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 300
#SBATCH -p sched_mit_rafagb,sched_opportunist
#SBATCH --mem-per-cpu=400
#SBATCH --no-requeue
#SBATCH --signal=B:2@300

source deactivate

source $HOME/.bashrc
export NFFDIR=/home/saxelrod/repo/nff/covid/NeuralForceField
export PYTHONPATH="$NFFDIR"

conda activate covid_mit

featurize=$HOME/repo/nff/covid/NeuralForceField/scripts/featurize.py
python  $featurize dataset.pth.tar

"""

# JOB_DIR = "/home/saxelrod/local_jobs"
JOB_DIR = "/home/saxelrod//engaging_nfs/jobs"
# D_PATH = "/home/saxelrod/engaging_nfs/data_from_fock/data/covid_data/covid_mmff94_1_50k.pth.tar"
D_PATH = "/home/saxelrod/engaging_nfs/data_from_fock/data/covid_data/all_crest.pth.tar"


def main(d_path=D_PATH, num_jobs=100, job_dir=JOB_DIR):

    dataset = Dataset.from_file(d_path)
    print("Splitting datasets...")
    datasets = split_dataset(dataset, num_jobs)
    print("Finished splitting datasets.")

    for d_set in datasets:

        import pdb
        pdb.set_trace()

        folder_name = "0000_featurize_" + str(uuid.uuid4())
        
        tmp_path = "/tmp/{}".format(folder_name)
        os.mkdir(tmp_path)
        d_set.save(os.path.join(tmp_path, "dataset.pth.tar"))
        with open(os.path.join(tmp_path, "job.sh"), "w") as f:
            f.write(TEMPLATE)

        real_path = os.path.join(job_dir, "inbox", folder_name)
        shutil.move(tmp_path, real_path)

        break

if __name__ == "__main__":
    main()

