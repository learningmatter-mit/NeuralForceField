import unittest
import numpy as np
import torch
from nff.data.dataset import Dataset, concatenate_dict 
from nff.data.stats import remove_dataset_outliers


class TestFunctions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = Dataset.from_file('../../tutorials/data/dataset.pth.tar')

    def test_concatenate(self):
        dict_1 = self.dataset[0]
        dict_2 = self.dataset[1:3]

        concat_dict = concatenate_dict(dict_1, dict_2)

        print(concat_dict['energy'])
        print(concat_dict['smiles'])


class TestStats(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = Dataset.from_file('../../tutorials/data/dataset.pth.tar')

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

        print(mean, std)
        print(np.mean(new_array), np.std(new_array))
        print(np.max(new_array), np.min(new_array))

if __name__ == '__main__':
    unittest.main()
