import os
from datetime import datetime
import json
import shutil

from nff.data import Dataset, split_train_validation_test

def get_path_names(save_dir, project_name):
    sigopt_path = os.path.join(save_dir, "sigopt")
    project_path = os.path.join(sigopt_path, project_name)

    return sigopt_path, project_path


def make_path(save_dir, project_name):
    """
    Structure:
        save_dir
            sigopt
                project_name
                    model_0
                    model_1
                    model_2
                    ...

    """

    # main folder

    sigopt_path, project_path = get_path_names(save_dir=save_dir,
                                               project_name=project_name)
    if not os.path.isdir(sigopt_path):
        os.mkdir(sigopt_path)

    # subfolder for a specific project
    if not os.path.isdir(project_path):
        os.mkdir(project_path)

    return sigopt_path, project_path


def make_model_folder(save_dir, project_name, model_id):

    make_path(save_dir=save_dir,
              project_name=project_name)
    sigopt_path, project_path = get_path_names(save_dir=save_dir,
                                               project_name=project_name)

    new_model_folder = os.path.join(project_path, "model_" + str(model_id))

    if os.path.isdir(new_model_folder):
        backup_folder = new_model_folder + "_backup"
        if os.path.isdir(backup_folder):
            shutil.rmtree(backup_folder)
        shutil.move(new_model_folder, backup_folder)
    os.mkdir(new_model_folder)

    return new_model_folder


def save_info(param_regime,
              assignments,
              value,
              metric_name,
              model_folder,
              sugg_id,
              expt_id,
              set_params):

    out_file = os.path.join(model_folder, "job_info.json")

    job_info = {"param_regime": param_regime,
                "set_params": set_params,
                "assignments": assignments,
                metric_name: value,
                "suggestion_id": sugg_id,
                "experiment_id": expt_id,
                "time": str(datetime.now())}

    with open(out_file, "w") as f:
        json.dump(job_info, f, indent=4, sort_keys=True)


def get_splits(dataset, val_size, test_size, save_dir, project_name):

    make_path(save_dir=save_dir,
              project_name=project_name)
    sigopt_path, project_path = get_path_names(save_dir=save_dir,
                                               project_name=project_name)

    data_names = ["train.pth.tar", "val.pth.tar", "test.pth.tar"]
    data_paths = [os.path.join(project_path, name) for name in data_names]
    if all([os.path.isfile(file) for file in data_paths]):
        datasets = (Dataset.from_file(file) for file in data_paths)
        return datasets

    datasets = split_train_validation_test(
        dataset,
        val_size=val_size,
        test_size=test_size)
    for name, dataset in zip(data_names, datasets):
        save_path = os.path.join(project_path, name)
        dataset.save(save_path)

    return datasets
