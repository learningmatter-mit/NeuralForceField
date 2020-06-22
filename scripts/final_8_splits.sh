#!/bin/bash
#SBATCH -p sched_mit_rafagb
#SBATCH -t 4300
#SBATCH -n 64
#SBATCH -N 1
#SBATCH --mem 350G

source deactivate
source activate covid_mit
rm /home/saxelrod/fock/final_covid_train/prop_sample.json

for i in {0,1,2,3,4,5,6,7}
do
cmd="python dataset_from_pickles.py --num_specs 294000 --num_threads 8 --thread "$i
echo $cmd
eval $cmd
done
