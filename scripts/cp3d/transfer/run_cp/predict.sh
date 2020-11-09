source deactivate
source ~/.bashrc
source activate nff

# change to your config files
CONFIG="config/schnet_feat_cov1/predict_config.json"

cmd="python predict.py --config_file $CONFIG"
echo $cmd
eval $cmd
