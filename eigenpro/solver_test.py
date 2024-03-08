import unittest

import torch
import scipy
import math

import eigenpro.solver as sol
import eigenpro.models.kernel_machine as km
import eigenpro.kernels as k
import eigenpro.preconditioners as pcd
import eigenpro.utils.device as dev
from eigenpro.utils.tensor import SingleDeviceTensor, BaseDeviceTensor, DistributedTensor
from eigenpro.utils.ops import gather

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
        self.kernel_fn = lambda x,z: k.laplacian(x,z,bandwidth=1.)
        self.device_manager = dev.DeviceManager(
                [torch.device('cpu'),
                # torch.device('cpu'),
                ]
            )

        self.data_X = torch.randn(self.n, self.d, dtype=self.dtype)
        self.data_Y = torch.randn(self.n, self.c, dtype=self.dtype)
        self.model_Z = torch.randn(self.p, self.d, dtype=self.dtype)
        self.model_W = torch.randn(self.p, self.c, dtype=self.dtype)


class TestEigenPro(unittest.TestCase, TestProblem):
    
    def setUp(self):
        unittest.TestCase.__init__(self)
        TestProblem.__init__(self)

        self.model = km.KernelMachine(
            self.kernel_fn,
            self.d, self.c, self.p,
            centers=self.model_Z, 
            weights=self.model_W,
            dtype=self.dtype,
            device_manager=self.device_manager
        )
        self.temporary_centers = torch.zeros(0, self.d, dtype=self.dtype)
        self.temporary_weights = torch.zeros(0, self.c, dtype=self.dtype)
        self.nystrom_centers_for_data = self.data_X[:self.sd]
        self.nystrom_weights_for_data = torch.zeros(self.sd, self.c, dtype=self.dtype)
        
        self.nystrom_center_ids_for_model = torch.arange(self.sm)
        self.nystrom_centers_for_model = self.model_Z[self.nystrom_center_ids_for_model]
        

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

        self.data_preconditioner = pcd.Preconditioner(
            kernel_fn = self.kernel_fn,
            centers = self.nystrom_centers_for_data,
            top_q_eig = self.qd
        )

        self.model_preconditioner = pcd.Preconditioner(
            kernel_fn = self.kernel_fn,
            centers = self.nystrom_centers_for_model,
            center_ids = self.nystrom_center_ids_for_model,
            top_q_eig = self.qm
        )

        self.solver = sol.EigenProSolver(
            model = self.model,
            data_preconditioner = self.data_preconditioner,
            model_preconditioner = self.model_preconditioner,
            dtype = self.dtype,
        )


    def run_iterator_step_by_batchIDs(self, bids, h):
        m = len(bids)
        lr = self.data_preconditioner.scaled_learning_rate(m)
        x = self.data_X[bids]
        y = self.data_Y[bids]
        g = (self.kernel_fn(x, self.model_Z) @ self.model_W
                + self.kernel_fn(x, self.temporary_centers) @ self.temporary_weights
                + self.kernel_fn(x, self.nystrom_centers_for_data) @ self.nystrom_weights_for_data
            ) - y

        ksg = self.kernel_fn(self.nystrom_centers_for_data, x) @  g 
        ftksg = self.Fd.T @ ksg
        fftksg = self.Fd @ ftksg

        self.nystrom_weights_for_data += lr*fftksg
        
        self.temporary_centers = torch.cat([self.temporary_centers, x])
        self.temporary_weights = torch.cat([self.temporary_weights, -lr*g])

        return h - lr*(self.kernel_fn(self.model_Z, x) @ g - self.Md @ ftksg)


    def test_T_steps_of_iterator(self, T=16):
        h = torch.zeros(self.p, self.c, dtype=self.dtype)
        if T > math.floor(self.p/self.mx):
            print(f'resetting T. was {T}. now is {math.floor(self.p/self.mx)}')
            T = math.floor(self.p/self.mx)
        torch.testing.assert_close(
            gather(self.solver.iterator.grad_accumulation), h)
        self.model.train()
        for t, batch_ids in enumerate(torch.randperm(self.n).split(self.mx)[:T]):
            h = self.run_iterator_step_by_batchIDs(batch_ids, h)
            self.solver.iterator.step(self.data_X[batch_ids], self.data_Y[batch_ids])
            torch.testing.assert_close(
                gather(self.solver.iterator.grad_accumulation), h)
            print(f"iteration: {t}")
        self.model.eval()
    

    def run_projector_step_by_batchIDs(self, bids, h):
        m = len(bids)
        lr = self.model_preconditioner.scaled_learning_rate(m)
        z = self.model_Z[bids]
        gz = h[bids]
        g = self.kernel_fn(z, self.model_Z) @ self.model_W - gz

        ksg = self.kernel_fn(self.nystrom_centers_for_model, z) @  g 
        ftksg = self.Fm.T @ ksg
        fftksg = self.Fm @ ftksg        

        self.model_W[bids] -= lr*g 
        self.model_W[self.nystrom_center_ids_for_model] += lr*fftksg


    def test_T_steps_of_projector(self, T=16):
        h = torch.randn(self.p, self.c, dtype=self.dtype)
        self.solver.projector.loader.dataset.data_y = h
        self.model.train()
        for t, batch_ids in enumerate(torch.arange(self.p).split(self.mz)[:T]):
            self.run_projector_step_by_batchIDs(batch_ids, h)
            self.solver.projector.step(h[batch_ids], batch_ids)
            torch.testing.assert_close(
                gather(self.model.weights), self.model_W)
            print(f"iteration: {t}")
        self.model.eval()


if __name__ == "__main__":
    unittest.main()
