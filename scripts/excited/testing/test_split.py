import os
import argparse
import json
from tqdm import tqdm

from nff.data import Dataset


def trim(dset, num_samples):
    dset.shuffle()
    new_props = {key: val[:num_samples] for
                 key, val in dset.props.items()}
    new_dset = Dataset(props=new_props, check_props=False)
    return new_dset


def make_copies(dset_path,
                new_dset_dir,
                num_samples_list,
                **kwargs):

    dset = Dataset.from_file(dset_path)
    new_dset_paths = []

    for num_samples in tqdm(num_samples_list):
        new_dset = trim(dset, num_samples)
        new_dset_path = os.path.join(new_dset_dir,
                                     f"{num_samples}_samples.pth.tar")
        new_dset.save(new_dset_path)
        new_dset_paths.append(new_dset_path)
    return new_dset_paths


def parse_job_info():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        help="file containing all details",
                        default='testing/copy_config.json')
    args = parser.parse_args()
    config_file = args.config_file
    with open(config_file, "r") as f_open:
        job_info = json.load(f_open)

    return job_info


def make_splits(split_config,
                new_dset_paths,
                new_dset_dir,
                num_samples_list,
                **kwargs):

    with open(split_config, "r") as f_open:
        info = json.load(f_open)

    for new_dset_path, num_samples in zip(new_dset_paths,
                                          num_samples_list):
        info["dset_path"] = new_dset_path
        model_path = os.path.join(new_dset_dir,
                                  f"splits_{num_samples}")
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        info["model_path"] = model_path

        with open(split_config, "w") as f_open:
            json.dump(info, f_open, indent=4)

        cmd = f"python ../split.py --config_file {split_config}"
        os.system(cmd)


def main():
    job_info = parse_job_info()
    new_dset_paths = make_copies(**job_info)
    make_splits(new_dset_paths=new_dset_paths,
                **job_info)


if __name__ == "__main__":
    main()
