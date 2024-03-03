"""Base class for solvers."""
import torch
import eigenpro.models.kernel_machine as km

class BaseSolver:

    def __init__(self, 
        model: km.KernelMachine, dtype: torch.dtype):
        
        self._model = model
        self._dtype = dtype

    @property
    def model(self):
        return self._model

    @property
    def dtype(self):
        return self._dtype

    def step(self):
        raise NotImplementedError("method not implemented in base class, must be implemented in subclass")

    def reset(self):
        raise NotImplementedError("method not implemented in base class, must be implemented in subclass")