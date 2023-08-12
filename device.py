import torch
from torch.cuda.comm import broadcast
from typing import Union

class Device():
    """Handles tensor operations across multiple devices."""

    def __init__(self, devices: list[torch.device]):
        """Initializes the Device object with available devices."""
        self.devices = devices

    def __call__(self, tensor: torch.Tensor, strategy: str = "broadcast") -> Union[torch.Tensor, list[torch.Tensor]]:
        """Applies a distribution strategy to a tensor."""
        if strategy == "divide_to_gpu":
            return self.distribute_tensor_across_gpus(tensor)
        elif strategy == "broadcast":
            return broadcast(tensor, self.devices)
        elif strategy == "base":
            return tensor.to(self.devices[0])

    def distribute_tensor_across_gpus(self, tensor: torch.Tensor) -> Llist[torch.Tensor]:
        """Divides a tensor evenly across multiple GPUs."""
        tensor_segments = []
        segment_size = tensor.shape[0] // len(self.devices)
        for i in range(len(self.devices)):
            if i < len(self.devices) - 1:
                tensor_slice = tensor[i * segment_size: (i + 1) * segment_size, :]
            else:
                tensor_slice = tensor[i * segment_size:, :]
            tensor_segments.append(tensor_slice.to(self.devices[i]))
        return tensor_segments

    @staticmethod
    def create() -> 'Device':
        """Creates a Device object representing all available GPUs or CPU if no GPUs are available."""
        if torch.cuda.is_available():
            device_list = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        else:
            device_list = [torch.device('cpu')]
        return Device(device_list)
