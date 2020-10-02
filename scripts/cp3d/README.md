# Running CP3D scripts

This folder contains scripts for doing ChemProp3D tasks. These include making a dataset out of pickle files with species information, and training a CP3D model using this dataset. Below are some details about how to use the different folders.


## Making a dataset
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

#### Reducing the number of conformers

If you've already made a dataset and you want to reduce the number of conformers, you can do that by running the script `scripts/cp3d/make_dset/trim_confs.sh`. The only arguments you need are:
- `from_model_path` (str): The old path to the model and dataset. The script assumes your datasets are in the folders `from_model_path/0`, `from_model_path/1`, ..., etc., as they would be if generated using the `dset_from_pickles.sh`.
- `to_model_path` (str) The path to the new dataset with fewer conformers
- `num_confs` (int): Number of conformers that you want in the new dataset

Of course you can always just run `dset_from_pickles.sh` again, but using fewer conformers. But running `trim_confs.sh` might save some time if your pickle files don't contain `rdmols`. In this case you'd have to generate `rdmols` from `xyz`'s all over again, which would not be worth it! 


## Training

- Coming soon...


## Hyperparameter optimization

- Coming soon...

## Transfer learning

There may be cases in which there are two similar regression/classification tasks, but one task has significantly more data than the other. This is the case, for example, with SARS-CoV-2 binding prediction. There is an available dataset with over 300,000 molecules for SARS-CoV binding, but only a few thousand for SARS-CoV-2 binding. In this case it can help to train a model on the larger dataset, then use the model to generate fixed fingerprints for the smaller dataset. These fixed fingerprints can then be used to train a model on the smaller set.

As detailed in our paper, we have found that using this transfer learning technique with CP3D yields significantly better results for SARS-CoV-2 binding than any 2D methods. 

Scripts are available for transfer learning `scripts/cp3d/transfer`. The main script is `transfer.sh`, which runs the following three scripts in order. Note that you can run any of the scripts on their own if you want (for example, if you've already made fingerprints, you can run scripts 2-3 below and skip 1).

1. The script `get_fps/make_fps.sh` takes the model and adds fingerprints to species in other datasets.

    - Details for the script are found in the file `get_fps/fp_config.json `. This is where you should change the values for your project. The keys in the file are:
        - `model_folder` (str): Folder in which the model is saved
        - `dset_folder` (str): Folder in which the dataset is saved. The dataset contains the data for which you want to add fingerprints.
        - `feat_save_folder` (str): Folder in which you want to save pickle files with the fingerprints.
        - `prop` (str): Name of the property that the model predicts, and that you want to save. If you just want to make fingerprints and don't want to the model predictions you can leave this unspecified.
        - `device` (str): Device you want to use when calling the model. Can be "cpu" or any integer from 0 to the number of GPUs you have available, if you want to use a GPU.
        - `batch_size` (int): Batch size when evaluating the model
        - `sub_batch_size` (int): Number of sub-batches you want to divide each batch into. For more than about 10 conformers per species, most GPUs won't be able to calculate fingerprints in parallel for all conformers of a species. 7 is usually a good choice for a typical GPU.
        - `test_only` (bool): Only create fingerprints and evaluate the model on the test set. This is useful if you just want the model predictions on the test set and aren't interested in transfer learning.
        - `add_sigmoid` (bool): Add a sigmoid layer at the end of the model. This should be done if you are doing classification and training with the BCELogits loss.
        - `metrics` (list[str]): The script will loop through each metric and use the model with the best validation score according to that metric. It will create fingerprints and predictions for all of the different models. Any metrics recorded during training can be used here, such as `auc`, `prc-auc`, `loss`, and `mae`.
        
        
2. The script `export_to_cp/save_feats.sh` exports the pickle files generated by `make_fps.sh` to a form readable by ChemProp.
    - Details for the script are in `export_to_cp/feat_config.json`. The keys are:
        - `feat_folder` (str): Folder in which you saved the pickle files from the model. Should be the same as `feat_save_folder` above.
        - `cp_save_folder` (str): Folder in which you want to save the exported ChemProp features.
        - `smiles_folder` (str): Folder in which the train, validation and test SMILES strings are saved. The files should be named `train_smiles.csv`, `val_smiles.csv`, and `test_smiles.csv`.
        - `metrics` (list[str]): Subset of the `metrics` above for which you want to export features to ChemProp.
        
3. The script `run_cp/run_all_tls.sh` runs ChemProp using the features generated above. It can loop through all CP3D models from which you generated features (e.g. model with the best `auc`, model with the best `prc`, etc.), and through combinations of training with features + MPNN and training with features alone. 
    - Details for the script are in `run_cp/all_tls_config.json`. The keys are:
        - `base_config_path` (str): Path to the config file for the ChemProp. All of these parameters will be used for the ChemProp runs, except if you specify `metrics`, `features_path` or `features_only`, as these will change as you loop through the combinations of `metrics`, `feat_options`, and `mpnn_options` below. 
        - `cp_folder` (str): Path to the ChemProp folder on your computer.
        - `feature_folder` (str): Folder in which the features you geneated above for ChemProp are saved. 
        - `model_folder_cp` (str): Folder in which you will make sub-folders for all your ChemProp models.
        - `metrics` (list[str]): Subset of the metrics above. For every metric there exists a corresponding best CP3D model and associated CP3D features. A new ChemProp model will be trained for every different set of these features. The associated ChemProp model will also be scored on that metric. 
        - `feat_options` (list[bool]): Whether to use the CP3D features in the ChemProp model. If you specify [True, False], then separate models will be trained, one in which they are used and one in which they aren't. This might be useful if you want to compare performance with and without 3D features.
        - `mpnn_options` (list[bool]):  Whether to use an MPNN in conjunction with the CP3D features in the ChemProp model. If you specify [True, False], then separate models will be trained, one in which an MPNN is used and one in which it isn't. 



