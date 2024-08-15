import torch
import eigenpro.kernels as kernels
import eigenpro.models.sharded_kernel_machine as skm
import eigenpro.solver as solver
import eigenpro.utils.device as dev
from eigenpro.data.utils import protein_2_1hot
from datasets import load_dataset

n_samples = 1000  # number of samples to use for training
n_centers = 100  # number of centers to use
n_classes = 2

dataset = load_dataset("proteinea/solubility")
dataset = dataset.filter(lambda x: len(x["sequences"]) > 100)


def process_sample(x):
    # binary label to one hot encoding
    binary_label = x["labels"]
    one_hot = [0, 0]
    one_hot[binary_label] = 1
    x["labels"] = one_hot
    # truncate sequences to 100
    x["sequences"] = x["sequences"][:100]
    x["aa_1hot"] = protein_2_1hot(x["sequences"])
    return x


dataset = dataset.map(process_sample)

dataset_train = dataset["train"].select(range(n_samples))
X_train = torch.tensor(dataset_train["aa_1hot"], dtype=torch.float32)
y_train = torch.tensor(dataset_train["labels"], dtype=torch.float32)

dataset_test = dataset["train"].select(range(n_samples, int(n_samples * 1.2)))
X_test = torch.tensor(dataset_test["aa_1hot"], dtype=torch.float32)
y_test = torch.tensor(dataset_test["labels"], dtype=torch.float32)

centers = X_train[torch.randperm(X_train.shape[0])[:n_centers]]

dtype = torch.float32


def kernel_fn(x, z):
    return kernels.hamming_imq(x1=x, x2=z, vocab_size=20, batch_shape=torch.Size([32]))


device = dev.Device.create(use_gpu_if_available=True)
model = skm.create_sharded_kernel_machine(
    centers, n_classes, kernel_fn, device, dtype=dtype, tmp_centers_coeff=2
)

sd, sm, qd, qm = 10, 10, 3, 3  # configuration for EigenPro preconditioners
model = solver.fit(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    device,
    dtype=dtype,
    kernel=kernel_fn,
    n_data_pcd_nyst_samples=sd,
    n_model_pcd_nyst_samples=sm,
    n_data_pcd_eigenvals=qd,
    n_model_pcd_eigenvals=qm,
    epochs=15,
    accumulated_gradients=True,
)
