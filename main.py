import numpy as np
import torch
import os
import wandb
import numpy as np
import ipdb
import argparse
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from device import Device
from run import run_eigenpro
from kernels import laplacian
from torchvision import datasets, transforms


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, help="Number of data points",default=50_000)
parser.add_argument("--p", type=int, help="Model size",default=20_000)
parser.add_argument("--s_data", type=int, help="s_data",default=2_000)
parser.add_argument("--s_model", type=int, help="s_model",default=2_000)
parser.add_argument("--q_data", type=int, help="q_data",default=100)
parser.add_argument("--q_model", type=int, help="q_model",default=100)
parser.add_argument("--epochs", type=int, help="q_model",default=2)
args = parser.parse_args()
print(args)


# Wandb: to use wandb you can use the template below and pass the `run` to `run_eigenpro` below
#######################################################################
# wandb_init = {}
# wandb_init["project_name"] ="---"
# wandb_init["mode"]  = 'online'
# wandb_init["key"]  = "----"
# wandb_init["org"] = "----"
# os.environ["WANDB_API_KEY"] = wandb_init['key']
# os.environ["WANDB_MODE"] = wandb_init['mode']  # online or offline
# run = wandb.init(project=wandb_init['project_name'], \
#                  entity=wandb_init['org'])
# run.name = f'---'
# run.save()
#######################################################################


# Loading data for FashionMNIST
data_root = './data/'
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)

n = args.n
p = args.p

X_all = train_dataset.data.reshape(-1, 28*28)/255.0
Y = one_hot(train_dataset.targets.long())
train_set_indices = np.random.choice(60_000, n, replace=False)
X = X_all[train_set_indices,:]
Y = Y[train_set_indices]

X_val = test_dataset.data.reshape(-1,28*28)/255.0
Y_val = one_hot(test_dataset.targets.long())


# Eigenpro configuration
dtype = torch.float16
n_epochs = args.epochs
s_data, s_model, q_data, q_model = args.s_data,  args.s_model, args.q_data, args.q_model
bandwidth = 20.0
kernel_fn = lambda x, z: laplacian(x, z, bandwidth=bandwidth)
# Note: if you want to run on CPU, change `dtype` to `torch.float32` since
# PyTorch does not support half-precision multiplication on CPU
device = Device.create(use_gpu_if_availabel=False)
device_base = device.device_base

# Eigenpro
# Note: if you want to use the whole X as your centers switch to EigenPro2.0 which is a faster method
use_all_x_as_centers = 0
if use_all_x_as_centers:
    model = run_eigenpro(X, X, Y, X_val, Y_val, device, type=dtype, kernel=kernel_fn,
                         s_data=s_data, s_model=s_model, q_data=q_data, q_model=q_model,
                         tmp_centers_coeff=2, wandb=None, epochs=n_epochs, accumulated_gradients=False)
else:
    # In case you want to use a subset of data as model centers, define Z as tensor of your centers
    centers_set_indices = np.random.choice(60_000, p, replace=False)
    Z = X_all[centers_set_indices,:]
    model = run_eigenpro(Z, X, Y, X_val, Y_val, device, type=dtype, kernel=kernel_fn,
                         s_data=s_data, s_model=s_model, q_data=q_data, q_model=q_model,
                         wandb=None, epochs=n_epochs)












































X,Y,Z  = load(dataset)
loader = Loader(X,y)

S_data = sample(X)
S_centers = sample(Z)

model = create_KernelModels(Z,n_outputs,kernel_fn) 
precon_data = Preconditioner(S_data,top_q_data)
precon_model = Preconditioner(S_centers,top_q_centers)
projection = Projector(Z)
T = 10
optim = Eigenpro(model,precon)

for t,x_batch,y_batch in enumerate(loader):
    optim.step(x_batch,y_batch)
    if (t+1)%T==0:
        projector(optim.model)
projector(optim.model)

