import torch
from torch.cuda.comm import broadcast
from typing import List, Union
from termcolor import colored


class Device:
    """Handles tensor operations across multiple devices."""

    def __init__(self, devices: List[torch.device]):
        """Initializes the Device object with available devices."""
        self.devices = devices
        self.device_base = devices[0]

    def __call__(
        self, tensor: torch.Tensor, strategy: str = "broadcast"
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Applies a distribution strategy to a tensor."""
        if strategy == "divide_to_gpu":
            return self.distribute_tensor_across_gpus(tensor)
        elif strategy == "broadcast":
            if len(self.devices) == 1 and self.devices[0] == torch.device("cpu"):
                return [tensor]
            else:
                return broadcast(tensor, self.devices)
        elif strategy == "base":
            return tensor.to(self.devices[0])

    def distribute_tensor_across_gpus(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Divides a tensor evenly across multiple GPUs."""
        tensor_segments = []
        segment_size = tensor.shape[0] // len(self.devices)
        for i in range(len(self.devices)):
            if i < len(self.devices) - 1:
                tensor_slice = tensor[i * segment_size : (i + 1) * segment_size, :]
            else:
                tensor_slice = tensor[i * segment_size :, :]
            tensor_segments.append(tensor_slice.to(self.devices[i]))
        return tensor_segments

    @staticmethod
    def create(use_gpu_if_available=True) -> "Device":
        """Creates a Device instance for all GPUs or CPU if there is no GPU."""
        if torch.cuda.is_available() and use_gpu_if_available:
            device_list = [
                torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
            ]
        else:
            device_list = [torch.device("cpu")]
        print(
            colored(
                f"notice: the current implementation can only support 1 "
                f"GPU, we only use the following device:"
                f" ({device_list[0]}) ",
                "red",
            )
        )
        return Device(device_list[0:1])
