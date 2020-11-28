source deactivate
source ~/.bashrc
source activate nff

# change to your config path
for i in $(seq 0 0); do

    CONFIG="config/cov_2_gen_cross_val/$i.json"
    cmd="python split.py --config_file $CONFIG"
    echo $cmd
    eval $cmd

done
