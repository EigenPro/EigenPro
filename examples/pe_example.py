import torch
import eigenpro.kernels as kernels
import eigenpro.models.sharded_kernel_machine as skm
import eigenpro.solver as solver
import eigenpro.utils.device as dev
from datasets import load_dataset

n, p, d, c = 1000, 100, 5, 2

dataset = load_dataset('large-scale-sequence-kernels/Proteinea_solubility_without_polyhist_tags_ZS_CLASS')

def mapper(x):
    # binary label to one hot encoding
    binary_label = x["target solubility"]
    one_hot = [0,0]
    one_hot[binary_label] = 1
    x["target solubility"] = one_hot
    return x
dataset = dataset.map(mapper)

dataset_train = dataset['whole_dataset'].select(range(n))
X_train = torch.tensor(dataset_train["aa_1hot"], dtype=torch.float32)
y_train = torch.tensor(dataset_train["target solubility"], dtype=torch.float32)

dataset_test = dataset['whole_dataset'].select(range(n, int(n*1.2)))
X_test = torch.tensor(dataset_test["aa_1hot"], dtype=torch.float32)
y_test = torch.tensor(dataset_test["target solubility"], dtype=torch.float32)

n_classes = 2

centers = X_train[torch.randperm(X_train.shape[0])[:p]]

dtype = torch.float32
def kernel_fn(x, z):
    x = x.cpu().numpy()
    z = z.cpu().numpy()
    return kernels.hamming_ker_imq(x, seqs_y=z, alphabet_name='prot', scale=1, beta=1/2, lag=1)
    # return kernels.laplacian(x, z, bandwidth=20.0)

device = dev.Device.create(use_gpu_if_available=True)
model = skm.create_sharded_kernel_machine(
    centers, n_classes, kernel_fn, device, dtype=dtype, tmp_centers_coeff=2)

sd, sm, qd, qm = 10, 10, 3, 3 # configuration for EigenPro preconditioners
model = solver.fit(model, X_train, y_train, X_test, y_test, device,
                   dtype=dtype, kernel=kernel_fn, n_data_pcd_nyst_samples=sd, n_model_pcd_nyst_samples=sm,
                   n_data_pcd_eigenvals=qd, n_model_pcd_eigenvals=qm, epochs=15, accumulated_gradients=True)
