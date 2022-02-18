source deactivate
source ~/.bashrc
source activate nff

# CONFIG="config/cov_2_cl_test.json"
CONFIG="config/uv_peaks.json"

cmd="python split.py --config_file $CONFIG"
echo $cmd
eval $cmd

