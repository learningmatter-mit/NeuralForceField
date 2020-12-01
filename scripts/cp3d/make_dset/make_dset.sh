#!/bin/bash
#SBATCH -N 8
#SBATCH -t 30240
#SBATCH --mem=300G
#SBATCH --no-requeue
#SBATCH --signal=B:2@300
#SBATCH --qos=high
#SBATCH -p normal
#SBATCH --constraint=xeon-g6


source deactivate
source ~/.bashrc

# change to your nff directory
export NFFDIR="$HOME/Repo/projects/covid_clean/NeuralForceField"
export PYTHONPATH="$NFFDIR:$PYTHONPATH"

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Interfacing with ChemProp to generate train/val/test labels for the dataset."
echo "-----------------------------------------------------------------------------------"
echo ""

cd splits
bash split.sh
cd - > /dev/null

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Using these splits together with the pickle files to generate a dataset."
echo "-----------------------------------------------------------------------------------"
echo ""

cd get_dset
bash dset_from_pickles.sh
cd - > /dev/null

