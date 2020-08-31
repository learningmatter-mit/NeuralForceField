source $HOME/.bashrc
export PYTHONPATH="/home/saxelrod/Repo/projects/covid_nff/NeuralForceField:${PYTHONPATH}"

source deactivate
source activate /home/saxelrod/miniconda3/envs/htvs

python cluster_fps.py --arg_path cluster_fps.json
