#!/bin/bash
#SBATCH -N 16
#SBATCH -t 30240
#SBATCH --mem=300G
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --ntasks-per-node 40
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:volta:2
#SBATCH --qos=high
#SBATCH -p normal
#SBATCH --constraint=xeon-g6

source $HOME/.bashrc
source deactivate
source activate nff

# find the conformer indices for all of the datasets

from_model_path="/home/gridsan/saxelrod/models/1095"
to_model_path="/home/gridsan/saxelrod/models/single_confs_from_mc"
dset_1="/home/gridsan/saxelrod/conf_mc_results/conf_train_update_loss_20_1_convgd.pth.tar"
fp_dset_path="/home/gridsan/saxelrod/models/4053/0_combined/all.pth.tar"
conf_file="/home/gridsan/saxelrod/models/1095/idx_from_update_loss_20.json"
batch_size="50"

cmd="python conf_idx_parallel.py --from_model_path $from_model_path "
cmd=$cmd" --dset_1_path $dset_1 --batch_size $batch_size "
cmd=$cmd"--fp_dset_path $fp_dset_path"

echo $cmd
# eval $cmd

# select these conformers and export a new dataset

cmd="python convert_data.py --from_model_path $from_model_path "
cmd=$cmd" --to_model_path $to_model_path --conf_file $conf_file"

echo $cmd
eval $cmd
