import numpy as np

import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets

def load_fmnist(data_root, n_train, n_test):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=transform)
    n_train_ = min(n_train, len(train_dataset))
    train_indices = np.random.choice(len(train_dataset), n_train_,
                                     replace=False)
    X_train = train_dataset.data[train_indices].reshape(-1, 28*28)/255.0
    Y_train = F.one_hot(train_dataset.targets[train_indices].long())
    
    test_dataset = datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=transform)
    n_test_ = min(n_test, len(test_dataset))
    test_indices = np.random.choice(len(test_dataset), n_test_, replace=False)
    X_test = test_dataset.data[test_indices].reshape(-1,28*28)/255.0
    Y_test = F.one_hot(test_dataset.targets[test_indices].long())

    return X_train, X_test, Y_train, Y_test
