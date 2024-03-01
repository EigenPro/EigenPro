import unittest

import torch
import scipy

import eigenpro.stateful_optimizers as opt
import eigenpro.models.sharded_kernel_machine as skm
import eigenpro.models.preallocated_kernel_machine as pkm
import eigenpro.models.kernel_machine as km
import eigenpro.kernels as k
import eigenpro.preconditioners as pcd
import eigenpro.utils.device as dev


class TestEigenPro(unittest.TestCase):
    
    def setUp(self):
        (self.n, self.d, self.c, self.p, 
         self.m, self.sm, self.sd, self.qm, self.qd
        ) = 1024, 16, 2, 512, 16, 256, 64, 32, 8
        
        self.dtype = torch.float64
        self.kernel_fn = lambda x,z: k.laplacian(x,z,bandwidth=1.)
        self.device_manager = dev.Device([torch.device('cpu')])

        self.data_X = torch.randn(self.n, self.d, dtype=self.dtype)
        self.data_y = torch.randn(self.n, self.c, dtype=self.dtype)
        self.model_Z = torch.randn(self.p, self.d, dtype=self.dtype)
        self.model_W = torch.randn(self.p, self.c, dtype=self.dtype)

        self.model = km.KernelMachine(
            kernel_fn = self.kernel_fn,
            n_inputs = self.d,
            n_outputs = self.c,
            size = self.p
        )
        self.model.centers = self.model_Z
        self.model.weights = self.model_W
        self.model.train()
        self.temporary_centers = torch.zeros(0, self.d, dtype=self.dtype)
        self.temporary_weights = torch.zeros(0, self.c, dtype=self.dtype)
        self.nystrom_centers = self.data_X[:self.sd]
        self.nystrom_weights = torch.zeros(self.sd, self.c, dtype=self.dtype)
        
        self.L, E = scipy.linalg.eigh(
            self.kernel_fn(self.nystrom_centers, self.nystrom_centers).numpy(),
            subset_by_index=[self.sd-self.qd-1, self.sd-1]
        )
        self.D = torch.from_numpy(1/self.L[1:]*(1-self.L[0]/self.L[1:])).flip(0)
        self.E = torch.from_numpy(E[:,1:]).fliplr() 
        self.F = self.E * self.D.sqrt()
        self.M = self.kernel_fn(self.model_Z, self.nystrom_centers) @ self.F

        self.data_preconditioner = pcd.Preconditioner(
            kernel_fn = self.kernel_fn,
            centers = self.nystrom_centers,
            top_q_eig = self.qd
        )

        self.model_preconditioner = pcd.Preconditioner(
            kernel_fn = self.kernel_fn,
            centers = self.model_Z[:self.sm],
            top_q_eig = self.qm
        )

        self.optimizer = opt.EigenPro(
            model = self.model,
            threshold_index = self.p,
            data_preconditioner = self.data_preconditioner,
            model_preconditioner = self.model_preconditioner,
            kz_xs_evecs = self.kernel_fn(self.model_Z, self.nystrom_centers) @ (self.E*self.D),
            dtype = self.dtype,
            tmp_centers_coeff = 2,
        )


    def predict(self, x):
        return (
            self.kernel_fn(x, self.model_Z) @ self.model_W
            + self.kernel_fn(x, self.temporary_centers) @ self.temporary_weights
            + self.kernel_fn(x, self.nystrom_centers) @ self.nystrom_weights
        )


    def add_temporary_centers(self, x, w):
        self.temporary_centers = torch.cat([self.temporary_centers, x])
        self.temporary_weights = torch.cat([self.temporary_weights, w])


    def run_step_by_ids(self, bids, h):
        m = len(bids)
        lr = self.data_preconditioner.scaled_learning_rate(m)
        x, y = self.data_X[bids], self.data_y[bids]
        g = self.predict(x) - y

        ksg = self.kernel_fn(self.nystrom_centers, x) @  g 
        ftksg = self.F.T @ ksg
        fftksg = self.F @ ftksg

        self.nystrom_weights += lr*fftksg
        self.add_temporary_centers(x, -lr*g)

        return h - lr*(self.kernel_fn(self.model_Z, x) @ g - self.M @ ftksg)


    def test_T_steps(self, T=16):
        h = torch.zeros(self.p, self.c, dtype=self.dtype)

        for t, batch_ids in enumerate(torch.randperm(self.n).split(self.m)[:T]):
            h = self.run_step_by_ids(batch_ids, h)
            self.optimizer.step(self.data_X[batch_ids], self.data_y[batch_ids], batch_ids)
            torch.testing.assert_close(self.optimizer.grad_accumulation, h)
            
    

    def run_projection_step(self):
        

    #     x1, y1, id1, x2, y2, id2, m, lr, g2, fftksg2, h = self.run_two_steps()
        



    # def test_third_step_with_projection(self):
    #     x1, y1, id1, x2, y2, id2, m, lr, g2, fftksg2, h = self.run_two_steps()
    #     x3, y3, id3 = self.data_X[2*m:3*m], self.data_y[2*m:3*m], torch.arange(2*m, 3*m)
        
    #     self.optimizer.step(x1, y1, id1, False)
    #     self.optimizer.step(x2, y2, id2, False)
    #     self.optimizer.step(x3, y3, id3, True)




if __name__ == "__main__":
    unittest.main()
