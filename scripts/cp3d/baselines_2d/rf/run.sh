source deactivate
source activate nff

export PYTHONPATH="/home/saxelrod/Repo/projects/master/NeuralForceField:$PYTHONPATH"
python run.py --config_file config/cov_2_cl.json
