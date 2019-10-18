import unittest
from nff.data.dataset import Dataset, concatenate_dict 

class TestFunctions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = Dataset.from_file('../../tutorials/data/dataset.pth.tar')

    def test_concatenate(self):
        dict_1 = self.dataset[0]
        dict_2 = self.dataset[1:3]

        concat_dict = concatenate_dict(dict_1, dict_2)


if __name__ == '__main__':
    unittest.main()
