#!/bin/bash
#SBATCH -N 1
#SBATCH -t 30240
##SBATCH --gres=gpu:2
#SBATCH --mem=300G
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 20
#SBATCH --gres=gpu:volta:2
#SBATCH --qos=high
#SBATCH -p normal
#SBATCH --constraint=xeon-g6

source deactivate
source activate nff

export NFFDIR="home/saxelrod/Repo/projects/covid_nff/NeuralForceField"
export PYTHONPATH="$NFFDIR:$PYTHONPATH"

source deactivate
source activate covid_mit
rm /home/saxelrod/fock/final_covid_train/prop_sample.json

for i in {0,1,2,3,4,5,6,7}
do
cmd="python dset_from_pickles.py --num_threads 8 --thread "$i
echo $cmd
eval $cmd
done

##### and then we need to run `regroup_datasets.py` if
# you happen to want to regroup them

