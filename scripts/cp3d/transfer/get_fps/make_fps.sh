#!/bin/bash
#SBATCH -N 8
#SBATCH -t 30240
#SBATCH --gres=gpu:2
#SBATCH --mem=300G
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 20
#SBATCH --gres=gpu:volta:2
#SBATCH --qos=high
#SBATCH -p normal
#SBATCH --constraint=xeon-g6

source $HOME/.bashrc
source activate nff

# change to your config path
CONFIG="config/cp3d_ndu_single_cov2_cl.json"
# CONFIG="config/bond_update_k1_single_cov1.json"

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
