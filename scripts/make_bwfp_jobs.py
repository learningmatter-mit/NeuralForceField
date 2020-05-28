import uuid
import os
import shutil

JOB_DIR = "/pool001/saxelrod/jobs"
# JOB_DIR = "testing"

TEMPLATE = """#!/bin/bash
#SBATCH -n 10
#SBATCH -N 1
#SBATCH -t 4300
#SBATCH -p sched_mit_rafagb,sched_opportunist
#SBATCH --mem=10G
#SBATCH --no-requeue
#SBATCH --signal=B:2@300

source $HOME/.bashrc
source activate htvs_4
make_bwfp=/home/saxelrod/repo/nff/covid/NeuralForceField/scripts/make_bwfp.py
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
python $make_bwfp {} 500 --num_confs 10


"""

def main(num_threads=100, job_dir=JOB_DIR):

    for thread in range(num_threads):

        folder_name = "0000_bwfp_" + str(uuid.uuid4())

        tmp_path = "/tmp/{}".format(folder_name)
        os.mkdir(tmp_path)
        with open(os.path.join(tmp_path, "job.sh"), "w") as f:
            f.write(TEMPLATE.format(thread))

        real_path = os.path.join(job_dir, "inbox", folder_name)
        shutil.move(tmp_path, real_path)

if __name__ == "__main__":
    main()
