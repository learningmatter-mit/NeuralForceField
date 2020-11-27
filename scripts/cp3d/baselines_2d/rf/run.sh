source deactivate
source activate nff

export PYTHONPATH="/home/saxelrod/Repo/projects/orb_net/NeuralForceField:$PYTHONPATH"
python run.py --config_file config.json
