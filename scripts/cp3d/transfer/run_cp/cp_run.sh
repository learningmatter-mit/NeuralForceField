source deactivate
source ~/.bashrc
source activate chemprop

model_0_path="$HOME/chemprop_cov_2/make_splits/cl_protease"

cwd="$HOME/chemprop_cov_2/make_splits/cl_protease"

###
cwd=$cwd"/hypopt"
###

metrics=("pr_auc" "loss" "roc_auc")
feat_options=("yes" "no")
mpnn_options=("yes" "no")

for features in ${feat_options[@]}; do
    for mpnn in ${mpnn_options[@]}; do
        for metric in ${metrics[@]}; do

            train_features="$model_0_path/chemprop_train_"$metric"_feats.csv"
            val_features="$model_0_path/chemprop_val_"$metric"_feats.csv"
            test_features="$model_0_path/chemprop_test_"$metric"_feats.csv"

            features_cmd=" --features_path $train_features --separate_val_features_path $val_features --separate_test_features_path $test_features"

            if [ "$features" = "yes" ]; then
                if [ "$mpnn" = "yes" ]; then
                    save_dir="$cwd/cp_feats_mpnn"
                    extra_flags=$features_cmd
                elif [ "$mpnn" = "no" ]; then
                    save_dir="$cwd/cp_feats_no_mpnn"
                     extra_flags="$features_cmd --features_only"
                fi
            else
                save_dir="$cwd/cp_just_mpnn"
                extra_flags=" "
            fi

            # cmd="python $HOME/Repo/projects/chemprop/train.py "
            ###
            cmd="python $HOME/Repo/projects/chemprop/hyperparameter_optimization.py "
            cmd=$cmd" --num_iters 100"
            ###

            cmd=$cmd" --data_path $HOME/chemprop_cov_2/make_splits/cl_protease/train_full.csv "
            cmd=$cmd" --separate_val_path $HOME/chemprop_cov_2/make_splits/cl_protease/val_full.csv "
            cmd=$cmd" --separate_test_path $HOME/chemprop_cov_2/make_splits/cl_protease/test_full.csv "
            cmd=$cmd" --dataset_type classification --no_features_scaling --quiet --gpu 2 --metric prc-auc --save_dir $save_dir "
            # cmd=$cmd"  --class_balance --epochs 300 --dropout 0.25 --depth 4 --ffn_num_layers 2 --hidden_size 500"
            # cmd=$cmd" --num_folds 10"
            cmd=$cmd" $extra_flags"

            echo $cmd
            eval $cmd

        done
    done
done
