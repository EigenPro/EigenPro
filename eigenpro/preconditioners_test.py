import unittest

import torch
import scipy

import eigenpro.preconditioners as pcd
import eigenpro.kernels as k


class TestPreconditioner(unittest.TestCase):

    def setUp(self):
        (self.n, self.d, self.c, self.p, 
         self.m, self.sm, self.sd, self.qm, self.qd
        ) = 1024, 16, 2, 512, 128, 256, 64, 32, 8
        
        self.dtype = torch.float64
        self.kernel_fn = lambda x,z: k.laplacian(x,z,bandwidth=1.)
        
        self.data_X = torch.randn(self.n, self.d)
        self.data_y = torch.randn(self.n, self.c)
        self.model_Z = torch.randn(self.p, self.d)
        self.nystrom_centers = self.data_X[:self.sd]

        self.preconditioner = pcd.Preconditioner(
            kernel_fn = self.kernel_fn,
            centers = self.nystrom_centers,
            top_q_eig = self.qd
        )

        self.L, E = scipy.linalg.eigh(
            self.kernel_fn(self.nystrom_centers, self.nystrom_centers).numpy(),
            subset_by_index=[self.sd-self.qd-1, self.sd-1]
        )
        self.D = torch.from_numpy(1/self.L[1:]*(1-self.L[0]/self.L[1:])).flip(0)
        self.E = torch.from_numpy(E[:,1:]).fliplr() 
        self.F = self.E * self.D.sqrt()
        

    def test_eigensys(self):
        torch.testing.assert_close(self.F, self.preconditioner.normalized_eigenvectors)

    def test_delta(self):
        grad = torch.randn(self.m, self.c)
        vtkg, vdvtkg = self.preconditioner.delta(
            self.data_X[:self.m], 
            grad
        )
        ksg = self.kernel_fn(self.nystrom_centers, self.data_X[:self.m]) @  grad 
        etksg = self.E.T @ ksg
        ftksg = self.F.T @ ksg
        fftksg = self.F @ ftksg
        torch.testing.assert_close(etksg, vtkg)
        torch.testing.assert_close(fftksg, vdvtkg)

if __name__ == "__main__":
    unittest.main()