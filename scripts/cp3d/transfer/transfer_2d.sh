source deactivate
source ~/.bashrc

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Making fingerprints using a pre-trained ChemProp model."
echo "Using the details specified in get_cp_fps/fp_cp_config.json."
echo "-----------------------------------------------------------------------------------"
echo ""

cd get_cp_fps
bash fps_from_cp.sh
cd - > /dev/null

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Training ChemProp models with different sets of features and architectures. "
echo "Using the details specified in run_cp/all_tls_config.json."
echo "-----------------------------------------------------------------------------------"
echo ""

cd run_cp
bash run_all_tls.sh

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Getting and saving the ChemProp predictions. "
echo "Using the details specified in run_cp/predict_config.json."
echo "-----------------------------------------------------------------------------------"
echo ""

bash predict.sh
cd - > /dev/null


