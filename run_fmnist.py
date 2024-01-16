import numpy as np
import torch
import os
import wandb
import numpy as np
import ipdb
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from eigenpro.utils.device import Device
from eigenpro.run import run_eigenpro
from eigenpro.kernels import laplacian
from eigenpro.utils import parse_cmd_args
from eigenpro.data.utils import load_fmnist

args = parse_cmd_args()

X_train, X_test, Y_train, Y_test = load_fmnist(os.environ["DATA_DIR"], args.n_train, args.n_test)

# Eigenpro configuration
dtype = torch.float32
kernel_fn = lambda x, z: laplacian(x, z, bandwidth=20.)
# Note: if you want to run on CPU, change `dtype` to `torch.float32` since
# PyTorch does not support half-precision multiplication on CPU
device = Device.create(use_gpu_if_available=False)
device_base = device.device_base

# Eigenpro
# Note: if you want to use the whole X as your centers switch to EigenPro2.0 which is a faster method
if args.model_size == -1:
    model = run_eigenpro(X_train, X_train, Y_train, X_test, Y_test, device, type=dtype, kernel=kernel_fn,
                         s_data=args.s_data, s_model=args.s_model, 
                         q_data=args.q_data, q_model=args.q_model,
                         tmp_centers_coeff=2, wandb=None, epochs=args.epochs, accumulated_gradients=False)
else:
    # In case you want to use a subset of data as model centers, define Z as tensor of your centers
    centers_set_indices = np.random.choice(args.n_train, args.model_size, replace=False)
    Z = X_train[centers_set_indices,:]
    model = run_eigenpro(Z, X_train, Y_train, X_test, Y_test, device, type=dtype, kernel=kernel_fn,
                         s_data=args.s_data, s_model=args.s_model, 
                         q_data=args.q_data, q_model=args.q_model,
                         wandb=None, epochs=args.epochs)















































