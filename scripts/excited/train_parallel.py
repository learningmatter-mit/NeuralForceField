"""
Train a model in parallel over multiple GPUs
"""

import subprocess
import os
import argparse
import json


def get_nodes():
    """
    If using slurm, get the nodes that the job is being run on.
    Args:
        None
    Returns:
        node_list (list): sorted list of the nodes
            being used.
    """

    cmd = "echo $(srun -l bash -c 'hostname' | sort | awk '{print $2}')"
    out = subprocess.check_output(
        cmd, env=os.environ.copy(), shell=True).decode()
    # e.g. returns "node1034.cm.cluster" for single node,
    # "node1115 node1116" for two nodes
    node_list = [item.split(".")[0] for item in out.split()]
    node_list = list(set(node_list))

    return node_list


def get_cpus():
    """
    Get the number of cpus per task.
    Args:
        None
    Returns:
        num_cpus (int): number of cpus per task
    """

    cpu_var = os.environ["SLURM_CPUS_PER_TASK"]
    num_cpus = int(cpu_var)

    return num_cpus


def get_gpus():
    """
    Get the number of gpus per node.
    Args:
        None
    Returns:
        num_gpus (int): number of gpus per node
    """

    gpu_var = os.environ["SLURM_GPUS_PER_NODE"]
    num_gpus = int(gpu_var)

    return num_gpus


def submit_slurm_job(node,
                     node_index,
                     num_gpus,
                     num_nodes,
                     cpus_per_task,
                     params_file):
    """
    Submit a job to a node using all of its available gpus.
    Args:
        node (str): name of node
        node_index (int): global node rank
        num_gpus (int): number of gpus per node
        num_nodes (int): total number of nodes
        cpus_per_task (int): number of cpus per task
        params_file (str): name of the file from which to load the
            nn params.
    Returns:
        p (subprocess): the process created by the submission.

    """

    cmd = (f"srun -N1 --nodelist {node} --ntasks 1  "
           f"--cpus-per-task {cpus_per_task} "
           f"python $NFFDIR/scripts/cp3d/train/train_single.py "
           f"{params_file} -nr {node_index} "
           f"--gpus {num_gpus} --nodes {num_nodes}")

    p = subprocess.Popen([cmd],
                         shell=True,
                         stdin=None,
                         stdout=None,
                         stderr=None,
                         close_fds=True)

    return p


def run_local_job(params_file,
                  num_gpus):
    """
    Run a (potentially parallel) job on a single node locally.
    Args:
        params_file (str): name of the file from which to load the
                nn params.
        num_gpus (int): number of GPUs
    Returns:
        p (subprocess): the process created by the submission.
    """

    cmd = (f"python $NFFDIR/scripts/cp3d/train/train_single.py "
           f"{params_file} -nr 0 "
           f"--gpus {num_gpus} --nodes 1")

    p = subprocess.Popen([cmd],
                         shell=True,
                         stdin=None,
                         stdout=None,
                         stderr=None,
                         close_fds=True)

    return p


def submit_to_nodes(params_file):
    """
    Submit jobs to all the nodes.
    Args:
        params_file (str): name of the file from which to load the
                nn params.
    Returns:
        None
    """

    with open(params_file, "r") as f:
        params = json.load(f)

    all_params = {**params["train_params"],
                  **params["model_params"]}
    use_slurm = all_params["use_slurm"]

    cmds = [("echo $(srun -l bash -c 'hostname' | sort | head -1 | "
             "awk '{print $2}')"),
            "echo $(getent hosts `hostname` | cut -d ' ' -f1)",
            "echo 8888"]

    env_vars = ["MASTER_ADDR", "MASTER_IP", "MASTER_PORT"]

    # if not using slurm, replace the first command with a local
    # command to get the host name

    if not use_slurm:
        cmds[0] = "echo $(cat /proc/sys/kernel/hostname)"

    # execute each of the commands

    for cmd, env_var in zip(cmds, env_vars):
        var = subprocess.check_output(
            cmd, env=os.environ.copy(), shell=True).decode()
        os.environ[env_var] = var

    if use_slurm:

        # get the number of nodes, cpus per task, and gpus per task

        node_list = get_nodes()
        cpus_per_task = get_cpus()
        num_gpus = get_gpus()

        num_nodes = len(node_list)
        procs = []

        # submit each node job

        for i, node in enumerate(node_list):
            p = submit_slurm_job(node=node,
                                 num_gpus=num_gpus,
                                 num_nodes=num_nodes,
                                 node_index=i,
                                 cpus_per_task=cpus_per_task,
                                 params_file=params_file)
            procs.append(p)

        # wait for them all to finish

        for p in procs:
            p.wait()

    else:

        # run a local job

        num_gpus = all_params["num_gpus"]
        p = run_local_job(params_file=params_file,
                          num_gpus=num_gpus)

        p.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('paramsfile', type=str,
                        help="file containing all parameters")

    arguments = parser.parse_args()

    submit_to_nodes(params_file=arguments.paramsfile)
