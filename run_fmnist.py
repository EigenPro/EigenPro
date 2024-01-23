import numpy as np
import torch
import os
import numpy as np
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from eigenpro.utils.device import Device
from eigenpro.run import run_eigenpro
from eigenpro.kernels import laplacian
from eigenpro.utils.cmd import parse_cmd_args
from eigenpro.data.utils import load_fmnist
from eigenpro.models import create_kernel_model

import time

args = parse_cmd_args()

X_train, X_test, Y_train, Y_test = load_fmnist(os.environ["DATA_DIR"], args.n_train, args.n_test)

# Eigenpro configuration
dtype = torch.float16
kernel_fn = lambda x, z: laplacian(x, z, bandwidth=20.)
# Note: if you want to run on CPU, change `dtype` to `torch.float32` since
# PyTorch does not support half-precision multiplication on CPU
device = Device.create(use_gpu_if_available=True)

start = time.time()
# Eigenpro
# Note: if you want to use the whole X as your centers switch to EigenPro2.0 which is a faster method
if args.model_size == -1:
    accumulated_gradients = False
    Z = X_train
else:
    # In case you want to use a subset of data as model centers, define Z as tensor of your centers
    accumulated_gradients = True
    centers_set_indices = np.random.choice(X_train.shape[0], args.model_size, replace=False)
    Z = X_train[centers_set_indices,:]

model = create_kernel_model(Z, Y_train.shape[-1], kernel_fn, device, dtype=dtype, tmp_centers_coeff=2)

model = run_eigenpro(model, X_train, Y_train, X_test, Y_test, device, dtype=dtype, kernel=kernel_fn,
                     s_data=args.s_data, s_model=args.s_model, 
                     q_data=args.q_data, q_model=args.q_model,
                     wandb=None, epochs=args.epochs,accumulated_gradients=accumulated_gradients)


end = time.time()

print(f'total time:{end-start}')
