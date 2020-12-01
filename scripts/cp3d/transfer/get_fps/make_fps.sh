#!/bin/bash
#SBATCH -N 1
#SBATCH -t 4300
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH -p sched_mit_rafagb_amd,sched_mit_rafagb

source $HOME/.bashrc
source activate nff

# change to your config path
CONFIG="config/cov2_cl_test.json"


export NFFDIR=/home/saxelrod/repo/nff/master/NeuralForceField
export PYTHONPATH="$NFFDIR:$PYTHONPATH"


metrics_lst=$(cat $CONFIG | jq ".metrics")

# get rid of the square brackets and commas
metric_str="${metrics_lst/[/}"
metric_str="${metric_str/]/}"
metric_str="${metric_str//,/ }"

# convert string to bash array
metrics=($metric_str)

echo $metric

for metric in ${metrics[@]}; do

	cmd="python make_fps.py --metric $metric --config_file $CONFIG "
	statement="Evaluating model using the $metric metric"
	echo $statement
	echo $cmd
	eval $cmd
	echo ""
done
