import os
import math
import numpy as np
import subprocess
import json
import argparse
import sys

from neuralnet.run_parallel import get_nodes


def fprint(msg):
    print(msg)
    sys.stdout.flush()


def split_resources(folders):

    cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])
    num_tasks = int(os.environ["SLURM_NTASKS_PER_NODE"])
    cpus_per_node = cpus_per_task * num_tasks

    node_list = get_nodes()
    num_nodes = len(node_list)

    node_alloc = np.array_split(folders, num_nodes)
    cpu_alloc = [math.floor(cpus_per_node / len(jobs))
                 for jobs in node_alloc]

    dics = []
    for i in range(len(node_alloc)):
        dic = {"folders": node_alloc[i],
               "num_cpus": cpu_alloc[i],
               "node": node_list[i]}
        dics.append(dic)

    return dics


def get_dic_save_path(dset_1_path, folder):
    name = dset_1_path.split("/")[-1].replace("_1_convgd.pth.tar", ""
                                              ).replace("conf_train_",
                                                        "")

    dic_save_path = os.path.join(str(folder), f"idx_from_{name}.json")

    return dic_save_path


def run_scripts(dset_1_path,
                folders,
                batch_size,
                fp_dset_path,
                update_fps,
                from_model_path):

	direc = os.path.abspath(".")
	script_path = os.path.join(direc, "get_conf_idx.py")

	dics = split_resources(folders)
	procs = []

	for dic in dics:
		these_folders = dic["folders"]
		# node = dic["node"]
		num_cpus = dic["num_cpus"]

		for folder in these_folders:

			folder_path = os.path.join(from_model_path, folder)
			dic_save_path = get_dic_save_path(dset_1_path, folder_path)

			py_cmd = (f"python {script_path} "
			          f"--folder_path {folder_path} "
			          f"--dset_1_path {dset_1_path} "
			          f"--dic_save_path {dic_save_path} "
			          f"--fp_dset_path {fp_dset_path} "
			          f"--batch_size {batch_size} ")
			if update_fps:
				py_cmd += "--update_fps"

			srun_cmd = (f"srun -N 1 --ntasks 1 "
			            f"--cpus-per-task {num_cpus} "
			            f"--exclusive {py_cmd}")

			p = subprocess.Popen([srun_cmd],
			                     shell=True,
			                     stdin=None,
			                     stdout=None,
			                     stderr=None,
			                     close_fds=True)

			fprint(srun_cmd)

			procs.append(p)

	fprint(f"Executed {len(procs)} commands. Waiting for results...")

	for p in procs:
		p.wait()


def combine_dics(folders, from_model_path, dset_1_path):

	overall_dic = {}
	for folder in folders:
		folder_path = os.path.join(from_model_path, folder)
		dic_save_path = get_dic_save_path(dset_1_path, folder_path)
		with open(dic_save_path, "r") as f:
			new_dic = json.load(f)
		overall_dic.update(new_dic)

	final_dic_path = get_dic_save_path(dset_1_path, from_model_path)
	with open(final_dic_path, "w") as f:
		json.dump(overall_dic, f)


def get_folders(from_model_path):
    folders = [i for i in os.listdir(from_model_path) if i.isdigit()]
    return folders


def main(dset_1_path,
         from_model_path,
         batch_size,
         fp_dset_path,
         update_fps,
         **kwargs):

    folders = get_folders(from_model_path)
    run_scripts(dset_1_path,
                folders,
                batch_size,
                fp_dset_path,
                update_fps,
                from_model_path)

    combine_dics(folders, from_model_path, dset_1_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_model_path', type=str,
                        help="Model with all the datasets")
    parser.add_argument('--fp_dset_path', type=str,
                        help="Reference dataset with fingerprints")
    parser.add_argument('--dset_1_path', type=str,
                        help="Path ConfDataset optimized from MC")
    parser.add_argument('--batch_size', type=int,
                        help=("Batch size with which to "
                              "find optimized conformers"),
                        default=50)
    parser.add_argument('--update_fps', action="store_true",
                        help=("Update datasets with fingerprints "
                              "even if they already have them"),
                        default=False)

    args = parser.parse_args()
    main(**args.__dict__)
