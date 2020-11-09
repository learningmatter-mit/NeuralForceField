source deactivate
source ~/.bashrc
source activate nff

# change to your config files
CONFIG="config/schnet_feat_cov1/all_tls_config.json"

cmd="python run_all_tls.py --config_file $CONFIG"
echo $cmd
eval $cmd
