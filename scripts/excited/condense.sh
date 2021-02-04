source deactivate
source ~/.bashrc
source activate nff

# change as necessary
export HTVSDIR="/home/saxelrod/repo/htvs/master/htvs"
export DJANGOCHEMDIR="/home/saxelrod/repo/htvs/master/htvs/djangochem"
export NFFDIR="/home/saxelrod/repo/nff/master/NeuralForceField"

python condense.py --config_file condense_config/schnet.json


