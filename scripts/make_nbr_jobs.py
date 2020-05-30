import uuid
import os
import shutil

# JOB_DIR = "/pool001/saxelrod/jobs"
JOB_DIR = "testing"


TEMPLATE = """#!/bin/bash
#SBATCH -n 10
#SBATCH -N 1
#SBATCH -t 4300
#SBATCH -p sched_mit_rafagb,sched_opportunist
#SBATCH --mem-per-cpu=1G
#SBATCH --no-requeue
#SBATCH --signal=B:2@300

source $HOME/.bashrc
source activate htvs_3
script=/home/saxelrod/repo/nff/covid/NeuralForceField/scripts/save_bond_nbrs
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
python $script --pickle_path {pickle_path} --save_path {save_path}


"""
BASE_PATH = "/pool001/saxelrod/engaging_nfs/data_from_fock/final_yaml"
NAME_START = "covid_fock_"
SUFFIX = ".pickle"


def main(base_path=BASE_PATH,
         name_start=NAME_START,
         job_dir=JOB_DIR,
         suffix=SUFFIX):

    files = []

    for file in os.listdir(base_path):
        if file.startswith(name_start) and file.endswith(suffix):
            files.append(file)

    for pickle_path in files:
        save_path = pickle_path.replace(suffix, "_nbrs" + suffix)
        text = TEMPLATE.format(pickle_path=pickle_path,
                               save_path=save_path)
        folder_name = "0000_nbrs_" + str(uuid.uuid4())

        tmp_path = "/tmp/{}".format(folder_name)
        os.mkdir(tmp_path)
        with open(os.path.join(tmp_path, "job.sh"), "w") as f:
            f.write(text)

        real_path = os.path.join(job_dir, "inbox", folder_name)
        shutil.move(tmp_path, real_path)


if __name__ == "__main__":
    main()
