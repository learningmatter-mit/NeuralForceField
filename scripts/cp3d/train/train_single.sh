source deactivate
source ~/.bashrc
source activate nff

# change to your config path
CONFIG="files/schnet_feat_cov1.json"

# change to your location of NeuralForceField
export NFFDIR="$HOME/repo/nff/covid_clean/NeuralForceField"
export PYTHONPATH=$NFFDIR:$PYTHON_PATH

cmd="python train_single.py $CONFIG -nr 0 --gpus 1 --nodes 1 & pid=\$!"
echo $cmd
eval $cmd
wait



