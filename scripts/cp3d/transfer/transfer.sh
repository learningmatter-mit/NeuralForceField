source deactivate
source ~/.bashrc

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Making fingerprints using a pre-trained model."
echo "Using the details specified in get_fps/fp_config.json."
echo "-----------------------------------------------------------------------------------"
echo ""

cd get_fps
bash make_fps.sh
cd - > /dev/null

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Saving the fingerprints to feature files for ChemProp."
echo "Using the details specified in export_to_cp/feat_config.json."
echo "-----------------------------------------------------------------------------------"
echo ""

cd export_to_cp
bash save_feats.sh
cd - > /dev/null

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Training ChemProp models with different sets of features and architectures. "
echo "Using the details specified in run_cp/all_tls_config.json."
echo "-----------------------------------------------------------------------------------"
echo ""

cd run_cp
bash run_all_tls.sh
cd - > /dev/null
