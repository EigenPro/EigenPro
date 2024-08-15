import numpy as np
import torch
import torch.utils.data as torch_data


class ArrayDataset(torch_data.Dataset):
    """A custom dataset to handle data along with their IDs and optional ID ranges."""

    def __init__(self, data_x, data_y, id_start=None, id_end=None):
        assert len(data_x) == len(data_y), "Data_x and Data_y must have the same length"

        if isinstance(data_x, np.ndarray):
            data_x = torch.tensor(data_x, dtype=torch.float32)
        if isinstance(data_y, np.ndarray):
            data_y = torch.tensor(data_y, dtype=torch.float32)

        self.data_x = data_x
        self.data_y = data_y

        # Check for negative ID start or end values
        if id_start is not None and id_start < 0:
            raise ValueError("id_start must be non-negative")
        if id_end is not None and id_end < 0:
            raise ValueError("id_end must be non-negative")

        # Check if id_start is greater than id_end
        if id_start is not None and id_end is not None and id_start >= id_end:
            raise ValueError("id_start must be less than id_end")

        if id_start is not None or id_end is not None:
            id_start = id_start if id_start is not None else 0
            id_end = id_end if id_end is not None else len(data_x)

            self.data_x = self.data_x[id_start:id_end]
            self.data_y = self.data_y[id_start:id_end]
            self.ids = list(range(id_start, id_end))
        else:
            self.ids = list(range(len(data_x)))

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return (
            self.data_x[index],
            self.data_y[index],
            torch.tensor(self.ids[index], dtype=torch.int64),
        )
