source deactivate
source ~/.bashrc
source activate nff

# change to your location of NeuralForceField
export NFFDIR="$HOME/repo/nff/covid_clean/NeuralForceField"
export PYTHONPATH=$NFFDIR:$PYTHON_PATH

python train_single.py single_config.json -nr 0 --gpus 1 --nodes 1 & pid=$!
wait



