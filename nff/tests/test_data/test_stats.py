import unittest
from pathlib import Path

import numpy as np
import torch

from nff.data.dataset import Dataset
from nff.data.stats import all_atoms, get_atom_count, remove_dataset_outliers

current_path = Path(__file__).parent

DATASET_PATH = current_path / "../../../tutorials/data/dataset.pth.tar"


class TestAtoms(unittest.TestCase):
    def test_get_atom_count(self):
        # Test case 1: Single atom formula
        formula = "H"
        expected_result = {"H": 1}
        self.assertEqual(get_atom_count(formula), expected_result)

        # Test case 2: Formula with multiple atoms
        formula = "H2O"
        expected_result = {"H": 2, "O": 1}
        self.assertEqual(get_atom_count(formula), expected_result)

        # Test case 3: Formula with repeated atoms
        formula = "CH3CH2CH3"
        expected_result = {"C": 3, "H": 8}
        self.assertEqual(get_atom_count(formula), expected_result)

    def test_all_atoms(self):
        unique_formulas = ["H2O", "CH4", "CO2"]
        expected_result = {"H", "O", "C"}

        result = all_atoms(unique_formulas)

        self.assertEqual(result, expected_result, "Incorrect atom set")


class TestStats(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset.from_file(DATASET_PATH)

    def test_remove_outliers_scalar(self):
        TEST_KEY = "energy"
        STD_AWAY = 1

        new_dset, _, _ = remove_dataset_outliers(self.dataset, reference_key=TEST_KEY, std_away=STD_AWAY)

        array = self.dataset.props[TEST_KEY].numpy()
        new_array = new_dset.props[TEST_KEY].cpu().numpy()

        ref_std = np.std(array)
        np.mean(array)

        assert np.max(new_array) - np.min(new_array) <= 2 * STD_AWAY * ref_std, "range is not working"

    def test_remove_outliers_tensor(self):
        TEST_KEY = "energy_grad"
        STD_AWAY = 3
        new_dset, _, _ = remove_dataset_outliers(self.dataset, reference_key=TEST_KEY, std_away=STD_AWAY)

        array = self.dataset.props[TEST_KEY]
        new_array = new_dset.props[TEST_KEY]

        stats_array = torch.cat(array, dim=0).flatten().cpu().numpy()

        ref_std = np.std(stats_array)
        np.mean(stats_array)

        new_stats_array = torch.cat(new_array, dim=0).flatten().cpu().numpy()

        assert np.max(new_stats_array) - np.min(new_stats_array) <= 2 * STD_AWAY * ref_std, "range is not working"


#        print(mean, std)
#        print(np.mean(new_array), np.std(new_array))
#        print(np.max(new_array), np.min(new_array))

if __name__ == "__main__":
    unittest.main()
