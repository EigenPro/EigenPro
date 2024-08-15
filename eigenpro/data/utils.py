import numpy as np

import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets

# More details about amino-acid sequences can be found at: https://en.wikipedia.org/wiki/Protein_primary_structure
AMINO_ACID_ALPHABET = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
AMINO_2_INT = dict((c, i) for i, c in enumerate(AMINO_ACID_ALPHABET))


def load_fmnist(data_root, n_train, n_test):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    n_train_ = min(n_train, len(train_dataset))
    train_indices = np.random.choice(len(train_dataset), n_train_, replace=False)
    X_train = train_dataset.data[train_indices].reshape(-1, 28 * 28) / 255.0
    Y_train = F.one_hot(train_dataset.targets[train_indices].long())

    test_dataset = datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    n_test_ = min(n_test, len(test_dataset))
    test_indices = np.random.choice(len(test_dataset), n_test_, replace=False)
    X_test = test_dataset.data[test_indices].reshape(-1, 28 * 28) / 255.0
    Y_test = F.one_hot(test_dataset.targets[test_indices].long())

    return X_train, X_test, Y_train, Y_test


def protein_2_1hot(sequence: str, flatten: bool = True) -> list[list[int]]:
    """Convert a protein sequence to one-hot encoding.

    Args:
        sequence : Protein sequence.
        flatten : Flatten the one-hot encoding. Defaults to True.

    Returns:
        One-hot encoding of the protein sequence
    """
    integer_encoded = [AMINO_2_INT[char] for char in sequence]
    onehot_encoded = list()

    for value in integer_encoded:
        letter = [0 for _ in range(len(AMINO_ACID_ALPHABET))]
        letter[value] = 1
        if flatten:
            onehot_encoded.extend(letter)
        else:
            onehot_encoded.append(letter)
    return onehot_encoded
