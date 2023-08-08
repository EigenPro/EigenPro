import torch
from torch.cuda.comm import broadcast


class Device():
    """Handles tensor operations across multiple devices."""

    def __init__(self, devices):
        """Initializes the Device object with available devices.

        Args:
            devices (list): List of available devices.
        """
        self.devices = devices

    def distribute_tensor_across_gpus(self, tensor):
        """Divides a tensor evenly across multiple GPUs.

        Args:
            tensor (torch.Tensor): Tensor to distribute.

        Returns:
            list: List of tensor segments, each located on a different device.
        """
        tensor_segments = []
        segment_size = tensor.shape[0] // len(self.devices)
        for i in range(len(self.devices)):
            if i < len(self.devices) - 1:
                tensor_slice = tensor[i * segment_size: (i + 1) * segment_size, :]
            else:
                tensor_slice = tensor[i * segment_size:, :]
            tensor_segments.append(tensor_slice.to(self.devices[i]))

        return tensor_segments

    def __call__(self, tensor, strategy="broadcast"):
        """Applies a distribution strategy to a tensor.

        Args:
            tensor (torch.Tensor): Tensor to distribute.
            strategy (str, optional): Distribution strategy. Defaults to "broadcast".

        Returns:
            list or torch.Tensor: Result of applying the distribution strategy to the tensor.
        """
        if strategy == "divide_to_gpu":
            return self.distribute_tensor_across_gpus(tensor)
        elif strategy == "broadcast":
            return broadcast(tensor, self.devices)
        elif strategy == "base":
            return tensor.to(self.devices[0])

    @staticmethod
    def create():
        """Creates a Device object representing all available GPUs or CPU if no GPUs are available.

        Returns:
            Device: Device object with all available GPUs or CPU.
        """
        if torch.cuda.is_available():
            device_list = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        else:
            device_list = [torch.device('cpu')]

        return Device(device_list)
