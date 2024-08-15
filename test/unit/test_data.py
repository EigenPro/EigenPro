import unittest
import pytest
import numpy as np
import torch
from eigenpro.data.array_dataset import ArrayDataset
from eigenpro.data.utils import protein_2_1hot


@pytest.fixture
def sequence():
    """The entire amino acid alphabet."""
    return "ACDEFGHIKLMNPQRSTVWY"


@pytest.mark.parametrize("flatten", [True, False])
def test_protein_2_1hot(sequence, flatten):
    """Since the input is the entire alphabet,
    the expected one-hot encoding should be an identity matrix of size 20.
    """
    one_hot = protein_2_1hot(sequence, flatten)
    if flatten:
        assert one_hot == np.eye(20).flatten().tolist()
    else:
        assert one_hot == np.eye(20).tolist()


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

    def test_tensor_input_handling(self):
        # Test handling of tensor inputs without converting them again
        data_x = torch.tensor([1.0, 2.0])
        data_y = torch.tensor([3.0, 4.0])
        dataset = ArrayDataset(data_x, data_y)
        self.assertIs(dataset.data_x, data_x)
        self.assertIs(dataset.data_y, data_y)

    def test_boundary_id_range(self):
        # Test boundary conditions for ID range
        dataset = ArrayDataset(
            np.array([1, 2, 3]), np.array([1, 2, 3]), id_start=0, id_end=3
        )
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.ids, [0, 1, 2])

    def test_negative_id_range(self):
        # Test handling of negative id_start and id_end
        with self.assertRaises(ValueError):
            ArrayDataset(
                np.array([1, 2, 3]), np.array([1, 2, 3]), id_start=-1, id_end=2
            )

    def test_id_start_greater_than_id_end(self):
        # Test handling when id_start is greater than id_end
        with self.assertRaises(ValueError):
            ArrayDataset(np.array([1, 2, 3]), np.array([1, 2, 3]), id_start=3, id_end=1)

    def test_mixed_input_types(self):
        # Test mixed numpy and tensor inputs
        data_x = np.array([1, 2, 3])
        data_y = torch.tensor([1, 2, 3], dtype=torch.float32)
        dataset = ArrayDataset(data_x, data_y)
        self.assertTrue(isinstance(dataset.data_x, torch.Tensor))
        self.assertTrue(isinstance(dataset.data_y, torch.Tensor))
        self.assertEqual(dataset.data_y.dtype, torch.float32)
