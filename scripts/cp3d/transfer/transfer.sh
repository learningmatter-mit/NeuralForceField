source deactivate
source ~/.bashrc
source activate nff

echo "Making fingerprints using a pre-trained model."
echo "Using the details specified in get_fps/fp_config.json."
bash get_fps/make_fps.sh

echo "Saving the fingerprints to feature files for ChemProp."
echo "Using the details specified in export_to_cp/feat_config.json."
bash export_to_cp/save_feats.sh

echo "Training ChemProp models with different sets of features and architectures. "
echo "Using the details specified in run_cp/all_tls_config.json."
bash run_cp/run_all_tls.sh

