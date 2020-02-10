import os
import unittest
import numpy as np
import torch
from nff.data.dataset import Dataset, concatenate_dict 
from nff.data.stats import remove_dataset_outliers

NFF_PATH = '../../../'
DATASET_PATH = os.path.join(NFF_PATH, 'tutorials/data/dataset.pth.tar')

class TestConcatenate(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = Dataset.from_file(DATASET_PATH)
        self.dict_a = {
            'a': 1,
            'b': 2
        }

        self.dict_a_list = {
            'a': [1],
            'b': [2]
        }

        self.dict_b = {
            'a': 3,
            'b': 4
        }

        self.dict_c = {
            'a': [5, 6],
            'b': [7, 8],
            'c': [9, 10]
        }

        self.dict_ab = {
            'a': [1, 3],
            'b': [2, 4],
        }

        self.dict_ac = {
            'a': [1, 5, 6],
            'b': [2, 7, 8],
            'c': [None, 9, 10]
        }

    def test_concat_1(self):
        ab = concatenate_dict(self.dict_a, self.dict_b)
        self.assertEqual(ab, self.dict_ab)

    def test_concat_2(self):
        ac = concatenate_dict(self.dict_a, self.dict_c)
        self.assertEqual(ac, self.dict_ac)

    def test_concat_single_dict(self):
        a = concatenate_dict(self.dict_a)
        self.assertEqual(a, self.dict_a_list)

    def test_concat_single_dict_lists(self):
        a = concatenate_dict(self.dict_a_list)
        self.assertEqual(a, self.dict_a_list)


class TestStats(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = Dataset.from_file(DATASET_PATH)

    def test_remove_outliers(self):
        TEST_KEY = 'energy'
        STD_AWAY = 1

        new_dset = remove_dataset_outliers(
            self.dataset,
            TEST_KEY,
            std_away=STD_AWAY
        )

        array = self.dataset.props[TEST_KEY].numpy()
        new_array = new_dset.props[TEST_KEY].cpu().numpy() 

        std = np.std(array)
        mean = np.mean(array)
        
        assert np.max(new_array) - np.min(new_array) \
            <= 2 * std, 'range is not working'

#        print(mean, std)
#        print(np.mean(new_array), np.std(new_array))
#        print(np.max(new_array), np.min(new_array))

if __name__ == '__main__':
    unittest.main()
