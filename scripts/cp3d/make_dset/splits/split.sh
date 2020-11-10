source deactivate
source ~/.bashrc
source activate nff

# change to your config path
CONFIG="config/cov_2_gen.json"
cmd="python split.py --config_file $CONFIG"
echo $cmd
eval $cmd
