for i in {0,1,2,3,4,5,6,7}
do
cmd="python dataset_from_pickles.py --num_specs 294000 --num_threads 8 --thread "$i
echo $cmd
eval $cmd
done
