source deactivate
source ~/.bashrc

# change to your nff directory
export NFFDIR="$HOME/Repo/projects/covid_clean/NeuralForceField"
export PYTHONPATH="$NFFDIR:$PYTHONPATH"

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Interfacing with ChemProp to generate train/val/test labels for the dataset."
echo "Using the details specified in splits/split_config.json."
echo "-----------------------------------------------------------------------------------"
echo ""

cd splits
bash split.sh
cd - > /dev/null

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Using these splits together with the pickle files to generate a dataset."
echo "Using the details specified in get_dset/dset_config.json."
echo "-----------------------------------------------------------------------------------"
echo ""

cd get_dset
bash dset_from_pickles.sh
cd - > /dev/null

