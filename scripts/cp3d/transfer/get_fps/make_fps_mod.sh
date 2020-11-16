#!/bin/bash
#SBATCH -N 8
#SBATCH -t 10000
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH -p sched_mit_rafagb_amd,sched_mit_rafagb

source $HOME/.bashrc
source activate nff_consistent

# change to your config path
CONFIG="config/schnet_feat_cov1_mod.json"

# change to your location of NeuralForceField
export NFFDIR="$HOME/repo/nff/master/NeuralForceField"
export PYTHONPATH=$NFFDIR:$PYTHON_PATH


# `jq` allows you to read a JSON file in bash. Here we are using it to get the number of threads from the config file.
#  If you don't have `jq` installed, you can download it by running `bash ../download_jq.sh`
metrics_lst=$(cat $CONFIG | $jq ".metrics")

# get rid of the square brackets and commas
metric_str="${metrics_lst/[/}"
metric_str="${metric_str/]/}"
metric_str="${metric_str//,/ }"

# convert string to bash array
metrics=($metric_str)

for metric in ${metrics[@]}; do

	cmd="python make_fps_mod.py --metric $metric --config_file $CONFIG "
	statement="Evaluating model using the $metric metric"
	echo $statement
	echo $cmd
	eval $cmd
	echo ""

done
