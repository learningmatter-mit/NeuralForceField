divider="---------------------------------"
divider=$divider$divider$divider

look_back=5

base_dir='/home/gridsan/saxelrod/models/switches'


echo $divider
echo "DIABATIC, 1.0 delta loss"
echo $divider

path=$base_dir'/painn_holdout_attention/log_human_read.csv'
echo $path
echo 'Slurm ID 9669716'
echo ''
cat $path | head -n 1
cat $path | tail -n $look_back


# echo $divider
# echo "*A*DIABATIC, 1.0 delta loss"
# echo $divider

# path=$base_dir'/painn_adiabatic/log_human_read.csv'
# echo $path
# echo 'Slurm ID 9669752'
# echo ''
# cat $path | head -n 1
# cat $path | tail -n $look_back


echo $divider
echo "DIABATIC, 0.5 delta loss"
echo $divider

path=$base_dir'/painn_holdout_attention_2/log_human_read.csv'
echo $path
echo 'Slurm ID 9669717'
echo ''
cat $path | head -n 1
cat $path | tail -n $look_back


echo $divider
echo "DIABATIC, 0.25 delta loss"
echo $divider

path=$base_dir'/painn_holdout_attention_3/log_human_read.csv'
echo $path
echo 'Slurm ID 9669718'
echo ''
cat $path | head -n 1
cat $path | tail -n $look_back


echo $divider
echo "*A*DIABATIC, 0.25 delta loss"
echo $divider

path=$base_dir'/painn_adiabatic_3/log_human_read.csv'
echo $path
echo 'Slurm ID 9669753'
echo ''
cat $path | head -n 1
cat $path | tail -n $look_back
