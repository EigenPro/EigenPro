import unittest

import torch
import scipy

import eigenpro.optimizers as opt
import eigenpro.models.sharded_kernel_machine as skm
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

        self.model = skm.create_sharded_kernel_machine(
            self.model_Z, self.c, self.kernel_fn, self.device_manager, dtype=self.dtype,
            tmp_centers_coeff=2
        )
        self.model.shard_kms[0]._weights[:self.p] = self.model_W
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


    def run_zero_steps(self):
        m = self.m
        lr = self.data_preconditioner.scaled_learning_rate(m)
        h = torch.zeros(self.p, self.c, dtype=self.dtype)
        return m, lr, h


    def test_zeroeth_step(self):

        m, lr, h = self.run_zero_steps()

        torch.testing.assert_close(self.optimizer.grad_accumulation, h)


    def run_one_step(self):
        
        m, lr, h = self.run_zero_steps()
        
        x1, y1, id1 = self.data_X[:m], self.data_y[:m], torch.arange(m)
        g1 = self.kernel_fn(x1, self.model_Z) @ self.model_W - y1

        ksg1 = self.kernel_fn(self.nystrom_centers, x1) @  g1 
        ftksg1 = self.F.T @ ksg1
        fftksg1 = self.F @ ftksg1
        mftksg1 = self.M @ ftksg1

        h -= lr* (self.kernel_fn(self.model_Z, x1) @ g1 - mftksg1)

        self.nystrom_weights += lr*fftksg1
        self.add_temporary_centers(x1, -lr*g1)

        return x1, y1, id1, m, lr, g1, fftksg1, h




    def test_first_step_without_projection(self):
        
        x1, y1, id1, m, lr, g1, fftksg1, h = self.run_one_step()
        
        self.optimizer.step(x1, y1, id1, False)
        
        torch.testing.assert_close(self.optimizer.grad_accumulation, h)


    def run_two_steps(self):

        x1, y1, id1, m, lr, g1, fftksg1, h = self.run_one_step()

        x2, y2, id2 = self.data_X[m:2*m], self.data_y[m:2*m], torch.arange(m, 2*m)
        g2 = self.predict(x2) - y2

        ksg2 = self.kernel_fn(self.nystrom_centers, x2) @  g2
        ftksg2 = self.F.T @ ksg2
        fftksg2 = self.F @ ftksg2
        mftksg2 = self.M @ ftksg2

        h -= lr* (self.kernel_fn(self.model_Z, x2) @ g2 - mftksg2)
        
        return x1, y1, id1, x2, y2, id2, m, lr, g2, fftksg2, h




    def test_second_step_without_projection(self):

        x1, y1, id1, x2, y2, id2, m, lr, g2, fftksg2, h = self.run_two_steps()
        
        self.optimizer.step(x1, y1, id1, False)
        self.optimizer.step(x2, y2, id2, False)

        torch.testing.assert_close(self.optimizer.grad_accumulation, h)
        

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
            
    

    # def run_projection_step(self):

    #     x1, y1, id1, x2, y2, id2, m, lr, g2, fftksg2, h = self.run_two_steps()
        



    # def test_third_step_with_projection(self):
    #     x1, y1, id1, x2, y2, id2, m, lr, g2, fftksg2, h = self.run_two_steps()
    #     x3, y3, id3 = self.data_X[2*m:3*m], self.data_y[2*m:3*m], torch.arange(2*m, 3*m)
        
    #     self.optimizer.step(x1, y1, id1, False)
    #     self.optimizer.step(x2, y2, id2, False)
    #     self.optimizer.step(x3, y3, id3, True)




if __name__ == "__main__":
    unittest.main()
