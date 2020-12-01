source deactivate
source ~/.bashrc
source activate nff

# change to your config path
export CONFIG="config/cov_2_cl_test.json"

cmd="python save_feats.py --config_file $CONFIG"
echo $cmd
eval $cmd
