source deactivate
source ~/.bashrc
source activate nff

# change to your config path
CONFIG="config/free_energy/cp_train_config.json"
cmd="python cp_train.py --config_file $CONFIG"

echo $cmd
eval $cmd

