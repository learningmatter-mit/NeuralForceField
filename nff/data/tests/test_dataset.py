import os
import unittest
import numpy as np
import torch
from nff.data.dataset import Dataset, concatenate_dict 
from nff.data.stats import remove_dataset_outliers

NFF_PATH = '../../../'
DATASET_PATH = os.path.join(NFF_PATH, 'tutorials/data/dataset.pth.tar')

class TestConcatenate(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset.from_file(DATASET_PATH)
        self.dict_a = {
            'a': 1,
            'b': 2,
            'str': 'abc'
        }

        self.dict_a_list = {
            'a': [1],
            'b': [2],
            'str': ['abc']
        }

        self.dict_b = {
            'a': 3,
            'b': 4,
            'str': 'efg'
        }

        self.dict_c = {
            'a': [5, 6],
            'b': [7, 8],
            'c': [9, 10],
            'str': ['aaa', 'bbb']
        }

        self.dict_d = {
            'a': [[1]],
            'b': [[2]],
        }

        self.dict_ab = {
            'a': [1, 3],
            'b': [2, 4],
            'str': ['abc', 'efg']

        }

        self.dict_ac = {
            'a': [1, 5, 6],
            'b': [2, 7, 8],
            'c': [None, 9, 10],
            'str': ['abc', 'aaa', 'bbb']
        }

        self.dict_dd = {
            'a': [[[1]], [[1]]],
            'b': [[[2]], [[2]]],
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

    def test_tensors(self):
        d1 = {'a': torch.tensor([1.])}
        d2 = {'a': torch.tensor([2., 3.])}
        dcat = concatenate_dict(d1, d2)
        expected = {
            'a': [
                torch.tensor(1.),
                torch.tensor(2.),
                torch.tensor(3.),
            ]
        }
        self.assertEqual(dcat, expected)

    def test_concat_list_lists(self):
        dd = concatenate_dict(self.dict_d, self.dict_d)
        self.assertEqual(dd, self.dict_dd)

    def test_concat_tensors(self):
        t = {
            'a': torch.tensor(1),
            'b': [torch.tensor([2, 3])],
            'c': torch.tensor([[1, 0], [0, 1]]),
        }
        tt = {
            'a': [torch.tensor(1)] * 2,
            'b': [torch.tensor([2, 3])] * 2,
            'c': [torch.tensor([[1, 0], [0, 1]])] * 2,
        }
        concat = concatenate_dict(t, t)
        for key, val in concat.items():
            for i, j in zip(val, tt[key]):
                self.assertTrue((i == j).all().item())

    def test_inexistent_list_lists(self):
        a = {
            'a': [[[1, 2]], [[3, 4]]], 
            'b': [5, 6]
        }
        
        b = {
            'b': [7, 8]
        }
        ab = concatenate_dict(a, b)
        expected = {
            'a': [
                [[1, 2]],
                [[3, 4]],
                None,
                None
            ], 
            'b': [5, 6, 7, 8]
        }
        self.assertEqual(ab, expected)


class TestStats(unittest.TestCase):
    def setUp(self):
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

class TestPeriodicDataset(unittest.TestCase):
    def setUp(self):
        self.quartz  = {
            "nxyz": np.array([
                [14.0, -1.19984241582007, 2.07818802527655, 4.59909615202747],
                [14.0, 1.31404847917993, 2.27599872954824, 2.7594569553608],
                [14.0, 2.39968483164015, 0.0, 0.919817758694137],
                [8.0, -1.06646793438585, 3.24694318819338, 0.20609293956337],
                [8.0, 0.235189576572621, 1.80712683722845, 3.8853713328967],
                [8.0, 0.831278357813231, 3.65430348422777, 2.04573213623004],
                [8.0, 3.34516925281323, 0.699883270597028, 5.31282465043663],
                [8.0, 1.44742296061415, 1.10724356663142, 1.6335462571033],
                [8.0, 2.74908047157262, 2.54705991759635, 3.47318545376996]
            ]),
            "lattice": np.array([
                [5.02778179, 0.0, 3.07862843796742e-16],
                [-2.513890895, 4.3541867548248, 3.07862843796742e-16],
                [0.0, 0.0, 5.51891759]
            ])
        }

        self.qtz_dataset = Dataset(concatenate_dict(*[self.quartz]*3))

    def test_neighbor_list(self):
        nbrs, offs = self.qtz_dataset.generate_neighbor_list(cutoff=5)


if __name__ == '__main__':
    unittest.main()
