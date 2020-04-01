# Overview
This is a module for optimizing NFF hyperparameters. To run `sigopt` to optimize the parameters, call
```
python run_sigopt.py <param_file.json>
```
where `<param_file.json>` is the name of the `JSON` file with the parameters you want to use. For an example file, see `info_files/covid_0.json`.

# Information to include
To see the most up-to-date list of all arguments that must be included, take a look at the `run_loop` function in `hyp_sigopt.py`.
Here is a list and explanation of the arguments to include:
- `project_name`: name of the project that you give to your hyperparameter optimization loop. Different folders get made
for different project names, so if you want to keep two loops separate, give them different project names.
- `param_regime`: List of dictionaries of the parameters for sigopt to optimize. Each dictionary contains the name of the
hyperparameter, its type, and its possible values.
- `set_params`: Parameters to include when building your network that are fixed (i.e. not optimized by sigopt).
- `budget`: Maximum number of hyperparameter optimization loops.
- `device`: Index of the GPU to use.
- `num_epochs`: Number of epochs to use in each hyperparameter optimization loop.
- `loss_name`: Name of the loss function to use in training.
- `target_name`: Name of the output that you want to optimize.
- `client_token`: Client token for sigopt
- `save_dir`: Directory in which to save different optimization loops
- `objective`: "minimize" or "maximize", depending on what your objective function is.
- `val_size`: Proportion of data to use in validation
- `test_size`: Proportion of data to use in testing
- `dataset_path`: Path to dataset to use for the loops
- `eval_metric`: Metric to use for evaluating the model performance
- `monitor_metrics`: Metrics to monitor during training
- `model_type`: Type of network you're training
- `model_kind`: Classification or regression
- `loss_coef`: Loss coefficient dictionary for training.
