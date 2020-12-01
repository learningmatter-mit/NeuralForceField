source deactivate
source ~/.bashrc
source activate nff

# change to your config files
CONFIG="config/cov2_test/predict_config.json"

cmd="python predict.py --config_file $CONFIG"
echo $cmd
eval $cmd
