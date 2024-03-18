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

from tests.instance import TestProblem

class TestModels(TestProblem, unittest.TestCase):
    def setUp(self):
        unittest.TestCase.__init__(self)
        TestProblem.__init__(self)

        
