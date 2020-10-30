source $HOME/.bashrc
source activate chemprop

cmd="python fps_from_cp.py --config_file fp_cp_config.json "
echo $cmd
eval $cmd
