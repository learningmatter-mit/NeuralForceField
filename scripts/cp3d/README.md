# Making a dataset
To make a CP3D dataset, we run the script `scripts/cp3d/make_dset/dset_from_pickles.sh`. This script can be used when you have a pickle file for each species. The pickle file should contain the following keys:

- `smiles`: The SMILES string of the species
- `conformers`: a list of dictionaries for each conformer
    - Each dictionary should contain at least the keys `boltzmannweight`, for the statistical weight of the conformer, and  `rd_mol`, an RDKit `mol` object for the conformer. 
    - If it doesn't contain `rd_mol` then it should contain `xyz` for the coordinates of the conformer. In this case the script will call `xyz2mol` to generate an RDKit `mol` object.
    
Details for the script are found in the file `dset_config.json`. This is where you should change the values for your project. The keys in the file are:
- `max_specs` (int): Maximum number of species to include in your dataset.
- `max_atoms` (int): Maximum number of atoms for a species that you will allow in your dataset. If you don't want a limit, set this value to None. 
- `max_confs` (int): Similar to `max_atoms`, but for the maximum number of conformers in a species.
- `nbrlist_cutoff` (float): cutoff for generating the 3D neighbor list
- `summary_path` (str): The path to the `JSON` file that contains a summary of the superset (the species from which you will sample to make your dataset). It has all the information about the species, excluding conformer/structural information. It also has the path to the corresponding pickle file, which has the structural information.
- `dataset_folder` (str): The path to the folder you want to save your dataset in.
- `pickle_folder` (str): The path to the folder that contains the pickle files.
- `prop_sample_path`: Path to the `JSON` file that contains all the info of `summary_path`, but also whether a species is in the train/val/test sets, or is not being used in the current dataset. 
- `num_threads` (int): How many files you want to divide your total dataset into. Conformer datasets can get quite large, so you may not want a single huge dataset that can't be loaded into memory. Instead you may want `num_threads` datasets. For example, if you perform parallel training with `N` GPUs, then you'll want to set `num_threads` to `N`. During train time, each parallel GPU thread will only read in one of these `N` datasets. For `m` GPUs per node, the node will only have to hold `m / N` of the total dataset in memory.



If you want to generate the splits yourself (e.g. with a ChemProp scaffold split), you can do so, and save to `prop_sample_path`. The script will then read the file. If you want the script to generate the splits itself, then the script can generate a split according to `sample_type`. 


- `sample_type` (str): Options are `random` and `classification`.
    -  If you want a random sample, then set `sample_type` to `random`. In this case you don't have to specify the keys below.
    - If you're doing a classification problem, you may want to generate a dataset whose proportion of positives is equal to the proportion of positives in the superset. To do this set `sample_type` to `classification`. In this case you'll also have to specify the keys below.

- `prop` (str): The name of the binary property that your model is predicting
- `pos_per_val` (int): number of positives that you want in the validation set
- `pos_per_test` (int): number of positives that you want in the test set
- `val_size` (int): absolute size of test set (i.e. number of species, not a proportion of the total dataset size)
- `test_size` (int): same, but for the test set




# Reducing the number of conformers

If you've already made a dataset and you want to reduce the number of conformers, you can do that by running the script `scripts/cp3d/trim_confs/trim_confs.sh`. The only arguments you need are:
- `from_model_path` (str): The old path to the model and dataset. The script assumes your datasets are in the folders `from_model_path/0`, `from_model_path/1`, ..., etc., as they would be if generated using the `dset_from_pickles.sh`.
- `to_model_path` (str) The path to the new dataset with fewer conformers
- `num_confs` (int): Number of conformers that you want in the new dataset

Of course you can always just run `dset_from_pickles.sh` again, but using fewer conformers. But running `trim_confs.sh` might save some time if your pickle files don't contain `rdmols`. In this case you'd have to generate `rdmols` from `xyz`'s all over again, which would not be worth it! 
