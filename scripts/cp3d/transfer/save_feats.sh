source deactivate
source activate /home/saxelrod/miniconda3/envs/htvs

cmd="python save_feats.py --model_name attention_k_1_yes_prob_cov_cl_protease "
cmd=$cmd" --path /home/saxelrod/supercloud2/models/cov_2_protease"
cmd=$cmd" --save_dir /home/saxelrod/chemprop_cov_2/features "

eval $cmd
