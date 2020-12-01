source deactivate
source ~/.bashrc

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Making fingerprints using a pre-trained model."
echo "-----------------------------------------------------------------------------------"
echo ""

cd get_fps
bash make_fps.sh
cd - > /dev/null

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Saving the fingerprints to feature files for ChemProp."
echo "-----------------------------------------------------------------------------------"
echo ""

cd export_to_cp
bash save_feats.sh
cd - > /dev/null

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Training ChemProp models with different sets of features and architectures. "
echo "-----------------------------------------------------------------------------------"
echo ""

cd run_cp
bash run_all_tls.sh

echo ""
echo "-----------------------------------------------------------------------------------"
echo "Getting and saving the ChemProp predictions. "
echo "-----------------------------------------------------------------------------------"
echo ""

bash predict.sh
cd - > /dev/null


