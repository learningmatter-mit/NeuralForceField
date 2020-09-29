import subprocess
import os
import argparse


def get_nodes():

    """
    Get the nodes that the job is being run on.
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


def submit_job(node, node_index, num_gpus, num_nodes, cpus_per_task,
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

    cmd = ("srun -N1 --nodelist {} --ntasks 1 --cpus-per-task {} "
           "python $NFFDIR/scripts/cp3d/train/train_nn.py "
           "{} -nr {} --gpus {} --nodes {}".format(
               node, cpus_per_task, params_file, node_index, num_gpus,
               num_nodes))

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

    # get the number of nodes, cpus per task, and gpus per task

    node_list = get_nodes()
    cpus_per_task = get_cpus()
    num_gpus = get_gpus()

    num_nodes = len(node_list)
    procs = []

    # submit each node job

    for i, node in enumerate(node_list):
        p = submit_job(node=node, num_gpus=num_gpus,
                       num_nodes=num_nodes, node_index=i,
                       cpus_per_task=cpus_per_task,
                       params_file=params_file)
        procs.append(p)

    # wait for them all to finish

    for p in procs:
        p.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('paramsfile', type=str,
                        help="file containing all parameters")

    arguments = parser.parse_args()

    submit_to_nodes(params_file=arguments.paramsfile)
