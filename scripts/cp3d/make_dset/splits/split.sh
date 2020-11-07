source deactivate
source ~/.bashrc
source activate chemprop

# change to your config path
CONFIG_FILE="files/cov_1.json"
cmd="python split.py --config_file $CONFIG_FILE"
echo $cmd
eval $cmd
