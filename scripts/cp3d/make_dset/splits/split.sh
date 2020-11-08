source deactivate
source ~/.bashrc
source activate chemprop

# change to your config path
CONFIG="config/cov_1.json"
cmd="python split.py --config_file $CONFIG"
echo $cmd
eval $cmd
