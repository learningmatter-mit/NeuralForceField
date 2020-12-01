source deactivate
source ~/.bashrc
source activate nff

# change to your config path
CONFIG="config/cov_2_gen_cp3d.json"

# change to your location of NeuralForceField
export NFFDIR="$HOME/repo/nff/master/NeuralForceField"
export PYTHONPATH=$NFFDIR:$PYTHON_PATH

cmd="python train_single.py $CONFIG -nr 0 --gpus 1 --nodes 1 " # & pid=\$!"
echo $cmd
eval $cmd
# wait



