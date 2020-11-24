source deactivate
source ~/.bashrc
source activate nff

# change to your config path
export CONFIG="config/schnet_feat_single_cov1.json"

cmd="python save_feats.py --config_file $CONFIG"
echo $cmd
eval $cmd
