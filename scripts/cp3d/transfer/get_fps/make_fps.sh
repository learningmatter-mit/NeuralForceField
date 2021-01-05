#!/bin/bash
#SBATCH -N 8
#SBATCH -t 30240
#SBATCH --mem=300G
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 20
#SBATCH --gres=gpu:volta:1
#SBATCH --qos=high
#SBATCH -p normal
#SBATCH --constraint=xeon-g6

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

# If using slurm parallel, delete any old
# prediction pickle files in the separate folders
# of the dataset path, as these will mess up the
# prediction.

dset_folder=$(echo $(cat $CONFIG | jq ".dset_folder") | sed -e 's/^"//' -e 's/"$//')
old_preds=$(ls $dset_folder/*/pred_*.pickle)
echo "Removing old files: $old_preds"
rm $old_preds

for metric in ${metrics[@]}; do

	cmd="python make_fps.py --metric $metric --config_file $CONFIG "
	statement="Evaluating model using the $metric metric"
	echo $statement
	echo $cmd
	eval $cmd
	echo ""
done
