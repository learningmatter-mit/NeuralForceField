source deactivate
source ~/.bashrc
source activate nff

# change to your config path
CONFIG="config/cp3d_cov2_cl_single.json"

# change to your location of NeuralForceField
export NFFDIR="$HOME/Repo/projects/master/NeuralForceField"
export PYTHONPATH=$NFFDIR:$PYTHON_PATH

cmd="python train_single.py $CONFIG -nr 0 --gpus 1 --nodes 1 "
echo $cmd
eval $cmd



