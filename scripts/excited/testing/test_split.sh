source deactivate
source activate nff

# change as necessary
THIS_HOME=$HOME
# THIS_HOME=$HOME/engaging

export HTVSDIR="$THIS_HOME/repo/htvs/master/htvs"
export DJANGOCHEMDIR="$THIS_HOME/repo/htvs/master/htvs/djangochem"
export NFFDIR="$THIS_HOME/repo/nff/master/NeuralForceField"


export PYTHONPATH=$HTVSDIR:$DJANGOCHEMDIR:$NFFDIR:$PYTHONPATH
export DJANGOCHEMDIR="djangochem.settings.orgel"
export DJANGO_SETTINGS_MODULE="djangochem.settings.orgel"

python test_split.py --config_file copy_config.json
