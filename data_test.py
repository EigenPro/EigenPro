import unittest
import numpy as np
import torch
from data import ArrayDataset

class TestArrayDataset(unittest.TestCase):

    def test_instantiation(self):
        # Test mismatched data lengths
        with self.assertRaises(AssertionError):
            ArrayDataset(np.array([1, 2]), np.array([1]))

        # Test numpy to tensor conversion
        dataset = ArrayDataset(np.array([1, 2]), np.array([1, 2]))
        self.assertTrue(isinstance(dataset.data_x, torch.Tensor))
        self.assertTrue(isinstance(dataset.data_y, torch.Tensor))

        # Test default ID range
        dataset = ArrayDataset(np.array([1, 2, 3]), np.array([1, 2, 3]))
        self.assertEqual(dataset.ids, [0, 1, 2])

        # Test custom ID range
        dataset = ArrayDataset(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), 1, 3)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.ids, [1, 2])

    def test_len_method(self):
        dataset = ArrayDataset(np.array([1, 2, 3]), np.array([1, 2, 3]))
        self.assertEqual(len(dataset), 3)

    def test_getitem_method(self):
        dataset = ArrayDataset(np.array([1, 2, 3]), np.array([4, 5, 6]))
        data_x, data_y, id_val = dataset[1]
        
        self.assertTrue(isinstance(data_x, torch.Tensor))
        self.assertTrue(isinstance(data_y, torch.Tensor))
        self.assertTrue(isinstance(id_val, torch.Tensor))
        
        self.assertEqual(data_x.item(), 2)
        self.assertEqual(data_y.item(), 5)
        self.assertEqual(id_val.item(), 1)

if __name__ == '__main__':
    unittest.main()
