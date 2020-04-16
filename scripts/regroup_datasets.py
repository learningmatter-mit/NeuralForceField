import sys
sys.path.insert(0, "..")

import os
from nff.data.dataset import Dataset
from nff.data.parallel import rejoin_props



JOB_DIR = "/home/saxelrod/engaging_nfs/jobs/completed"

SAVE_PTH = ("/home/saxelrod/engaging_nfs/data_from_fock/data/"
            "covid_data/covid_mmff94_1_50k_features.pth.tar")


def main(job_dir=JOB_DIR, save_pth=SAVE_PTH):

    datasets = []
    print("Loading datasets...")

    for i, folder in enumerate(os.listdir(job_dir)):
        folder_path = os.path.join(job_dir, folder)
        if not os.path.isdir(folder_path) or (
                'featurize' not in folder_path):
            continue
        path = os.path.join(folder_path, "dataset.pth.tar")
        dataset = Dataset.from_file(path)
        datasets.append(dataset)

    print("Loaded datasets.")
    print("Rejoining datasets...")

    dataset = datasets[0].copy()
    new_props = rejoin_props(datasets)
    dataset.props = new_props

    print("Rejoined datasets. New dataset has size %d"
          % (len(dataset)))

    print("Saving new dataset...")

    dataset.save(SAVE_PTH)

    print("Saved new dataset.")


if __name__ == "__main__":
    main()
