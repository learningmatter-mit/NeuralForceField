source deactivate
source activate nff


export DJANGOCHEMDIR="djangochem.settings.orgel"
export DJANGO_SETTINGS_MODULE="djangochem.settings.orgel"

python split.py --config_file split_info.json
