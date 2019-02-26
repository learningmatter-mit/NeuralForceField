#!/bin/bash
#SBATCH --partition=sched_mit_rafagb      # Partition to submit to
#SBATCH -n 12                             # Number of cores           
#SBATCH -N 1                              # Number of nodes
#SBATCH -t 60:00:00                       # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5000                # Memory (MB of RAM) per core (see also --mem)
#SBATCH --gres=gpu:1                      # Number of GPUs (per node)
#SBATCH -o slurm_output.out               # File to which STDOUT will be written
#SBATCH -e slurm_error.err                # File to which STDERR will be written
#SBATCH --nodelist=node1034

# Submit this job manually by running 'sbatch job.sh' from the login node 
# (if you are not using the htvs job manager, which you should be)

# Prepare your environment
source /home/wwj/.bashrc
source activate /home/wwj/anaconda3/envs/MPFF

# Below are some commands you can use to look into the activity of the GPU(s)
#lspci | grep -i nvidia
#nvidia-smi
#which nvidia-smi
#nvcc -V

# Execute the script
python /home/wwj/experiments/projects/NeuralForceField/train.py par.json
