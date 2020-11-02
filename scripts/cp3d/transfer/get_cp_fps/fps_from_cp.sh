source $HOME/.bashrc
source activate nff

cmd="python fps_from_cp.py --config_file fp_cp_config.json "
echo $cmd
eval $cmd
