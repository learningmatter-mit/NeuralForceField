source deactivate
source activate nff
# source activate htvs

# change as necessary
# export HTVSDIR="/home/saxelrod/repo/htvs/master/htvs"
# export DJANGOCHEMDIR="/home/saxelrod/repo/htvs/master/htvs/djangochem"
# export NFFDIR="/home/saxelrod/repo/nff/master/NeuralForceField"

# THIS_HOME=$HOME
THIS_HOME=$HOME/engaging

export HTVSDIR="$THIS_HOME/repo/htvs/master/htvs"
export DJANGOCHEMDIR="$THIS_HOME/repo/htvs/master/htvs/djangochem"
export NFFDIR="$THIS_HOME/repo/nff/master/NeuralForceField"

# export HTVSDIR="/home/saxelrod/htvs"
# export DJANGOCHEMDIR="/home/saxelrod/htvs/djangochem"
# export NFFDIR="/home/saxelrod/Repo/projects/master/NeuralForceField"


export PYTHONPATH=$HTVSDIR:$DJANGOCHEMDIR:$NFFDIR:$PYTHONPATH

export DJANGOCHEMDIR="djangochem.settings.orgel"
export DJANGO_SETTINGS_MODULE="djangochem.settings.orgel"

# CONFIG="split_info.json"
# CONFIG="split_info_test.json"
CONFIG="split_fock.json"

python split.py --config_file $CONFIG
