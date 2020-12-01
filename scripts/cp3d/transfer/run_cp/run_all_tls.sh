source deactivate
source ~/.bashrc
source activate nff

# change to your config files
CONFIG="config/cov2_test/all_tls_config.json"

cmd="python run_all_tls.py --config_file $CONFIG"
echo $cmd
eval $cmd
