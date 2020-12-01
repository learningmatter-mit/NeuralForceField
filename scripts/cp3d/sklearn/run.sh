source deactivate
source activate nff

export PYTHONPATH="/home/saxelrod/Repo/projects/master/NeuralForceField:$PYTHONPATH"
CONFIG="config/rf/cov_2_cl_auc.json"

cmd="python run.py --config_file $CONFIG"
echo $cmd
eval $cmd
