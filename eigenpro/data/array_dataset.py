import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ArrayDataset(Dataset):
    """A custom dataset to handle data along with their IDs and optional ID ranges."""

    def __init__(self, data_x, data_y, id_start=None, id_end=None):
        """
        Args:
        data_x (numpy.ndarray or torch.Tensor): Features data.
        data_y (numpy.ndarray or torch.Tensor): Labels data.
        id_start (int, optional): Starting ID. Defaults to None.
        id_end (int, optional): Ending ID (exclusive). Defaults to None.
        """
        assert len(data_x) == len(data_y), "Data_x and Data_y must have the same length"

        # Convert numpy arrays to tensors
        if isinstance(data_x, np.ndarray):
            data_x = torch.tensor(data_x, dtype=torch.float32)
        if isinstance(data_y, np.ndarray):
            data_y = torch.tensor(data_y, dtype=torch.float32)

        self.data_x = data_x
        self.data_y = data_y

        # If ID range is provided, slice the data accordingly
        if id_start is not None or id_end is not None:
            id_start = id_start if id_start is not None else 0
            id_end = id_end if id_end is not None else len(data_x)

            self.data_x = self.data_x[id_start:id_end]
            self.data_y = self.data_y[id_start:id_end]
            self.ids = list(range(id_start, id_end))
        else:
            self.ids = list(range(len(data_x)))

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data_x)

    def __getitem__(self, index):
        """
        Args:
        index (int): Index of the sample.

        Returns:
        tuple: (sample feature, sample label, sample ID)
        """
        return self.data_x[index], self.data_y[index], torch.tensor(self.ids[index], dtype=torch.int64)
