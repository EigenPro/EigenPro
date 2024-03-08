import unittest

import torch
import scipy
import math


class TestProblem:
    def __init__(self):
        self.n = 1024
        self.d = 16
        self.c = 2 
        self.p = 512
        self.mx = 128
        self.mz = 4
        self.sm = 256
        self.sd = 64
        self.qm = 32
        self.qd = 8
        
        self.dtype = torch.float64

        self.data_X = torch.randn(self.n, self.d, dtype=self.dtype)
        self.data_Y = torch.randn(self.n, self.c, dtype=self.dtype)
        self.model_Z = torch.randn(self.p, self.d, dtype=self.dtype)
        self.model_W = torch.randn(self.p, self.c, dtype=self.dtype)

        self.temporary_centers = torch.zeros(0, self.d, dtype=self.dtype)
        self.temporary_weights = torch.zeros(0, self.c, dtype=self.dtype)
        self.nystrom_centers_for_data = self.data_X[:self.sd]
        self.nystrom_weights_for_data = torch.zeros(self.sd, self.c, dtype=self.dtype)
        
        self.nystrom_center_ids_for_model = torch.arange(self.sm)
        self.nystrom_centers_for_model = self.model_Z[self.nystrom_center_ids_for_model]
        
        
    def calculate_eigensystems(self):
        Ld, Ed = scipy.linalg.eigh(
            self.kernel_fn(self.nystrom_centers_for_data, self.nystrom_centers_for_data).numpy(),
            subset_by_index=[self.sd-self.qd-1, self.sd-1]
        )
        Dd = torch.from_numpy(1/Ld[1:]*(1-Ld[0]/Ld[1:])).flip(0)
        Ed = torch.from_numpy(Ed[:,1:]).fliplr() 
        self.Fd = Ed * Dd.sqrt()
        self.Md = self.kernel_fn(self.model_Z, self.nystrom_centers_for_data) @ self.Fd


        Lm, Em = scipy.linalg.eigh(
            self.kernel_fn(self.nystrom_centers_for_model, self.nystrom_centers_for_model).numpy(),
            subset_by_index=[self.sm-self.qm-1, self.sm-1]
        )
        Dm = torch.from_numpy(1/Lm[1:]*(1-Lm[0]/Lm[1:])).flip(0)
        Em = torch.from_numpy(Em[:,1:]).fliplr() 
        self.Fm = Em * Dm.sqrt()