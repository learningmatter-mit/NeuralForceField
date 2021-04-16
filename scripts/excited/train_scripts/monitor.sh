divider="---------------------------------"
divider=$divider$divider$divider

look_back=5

base_dir='/home/gridsan/saxelrod/models/switches'


# echo $divider
# echo "Q-Chem TL"
# echo $divider

# path=$base_dir'/qchem_tl/log_human_read.csv'
# echo $path
# echo 'Slurm ID 9866970'
# echo ''
# cat $path | head -n 1
# cat $path | tail -n $look_back



echo $divider
echo "Q-Chem no TL"
echo $divider

path=$base_dir'/qchem_non_tl/log_human_read.csv'
echo $path
echo 'Slurm ID 9866976'
echo ''
cat $path | head -n 1
cat $path | tail -n $look_back



echo $divider
echo "Q-Chem no TL, nacv loss 3"
echo $divider

path=$base_dir'/qchem_non_tl_nacv_3/log_human_read.csv'
echo $path
echo 'Slurm ID 9873454'
echo ''
cat $path | head -n 1
cat $path | tail -n $look_back





echo $divider
echo "Q-Chem no TL no NACV"
echo $divider

path=$base_dir'/qchem_non_tl_no_nacv/log_human_read.csv'
echo $path
echo 'Slurm ID 9870836'
echo ''
cat $path | head -n 1
cat $path | tail -n $look_back


echo $divider
echo "GAMESS with Q-Chem geoms"
echo $divider

path=$base_dir'/gamess_w_qchem_geoms/log_human_read.csv'
echo $path
echo 'Slurm ID 9872289'
echo ''
cat $path | head -n 1
cat $path | tail -n $look_back



