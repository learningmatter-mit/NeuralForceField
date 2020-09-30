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

model_path="/home/gridsan/saxelrod/models"
# model_name="attention_k_1_no_prob_cov_cl_protease"
# model_name="attention_k_1_yes_prob_cov_cl_protease"
model_name="single_geom"

# base_cmd="python eval_test.py --model_path $model_path/1092 "
base_cmd="python eval_test.py --model_path $model_path/saved_models/single_geom "
base_cmd=$base_cmd" --gpu 0 --batch_size 1 --all_splits "

# paths=("cov_2_protease" "cov_2_all")
paths=("cov_2_protease")

metrics=("loss" "prc_auc" "roc_auc")
# metrics=("loss")

for path in ${paths[@]}; do
	for metric in ${metrics[@]}; do
		save_path="$model_path/$path/$model_name"
		cmd=$base_cmd" --path $model_path/$path --metric $metric "
		cmd=$cmd" --save_path $save_path "
		statement="Evaluating model using the $metric metric for the $path dataset"
		echo $statement
		echo $cmd
		eval $cmd
		echo ""

	done
done
