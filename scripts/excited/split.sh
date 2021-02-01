source deactivate
source activate htvs

# change as necessary
export HTVSDIR="/home/saxelrod/repo/htvs/master/htvs"
export DJANGOCHEMDIR="/home/saxelrod/repo/htvs/master/htvs/djangochem"
export NFFDIR="/home/saxelrod/repo/nff/master/NeuralForceField"
export PYTHONPATH=$HTVSDIR:$DJANGOCHEMDIR:$NFFDIR:$PYTHONPATH

export DJANGOCHEMDIR="djangochem.settings.orgel"
export DJANGO_SETTINGS_MODULE="djangochem.settings.orgel"

# CONFIG="split_info.json"
CONFIG="split_info_test.json"

python split.py --config_file $CONFIG
