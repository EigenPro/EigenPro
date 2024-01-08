from .eigenpro import FKR_EigenPro

from .kernel import gaussian, laplacian,laplacian_split
from torch.nn.functional import one_hot
import torchvision
import ipdb
import numpy as np
import torch

def vanilaep(kernel_fn,x_train,y_train,x_test,y_test,device,iters=10,reuse_batch=False):
    n_class = y_train.shape[1]
    model = FKR_EigenPro(kernel_fn, x_train, n_class, device=device)
    result = model.fit(x_train, y_train, x_test, y_test, epochs=list(range(iters)), mem_gb=30,reuse_batch=reuse_batch)
    return result

if __name__ == "__main__":
    devices = ['cuda']
    # dataset = Cifar5mmobilenet256augment(subsample=256_000, num_knots=100, device=devices,
    #                                      knot_include=0)
    # knots_x, knots_y = dataset.knots_x, one_hot(dataset.knots_y)
    # test_forEP2,testy = dataset.X_test,dataset.y_test
    # del dataset

    trainset = torchvision.datasets.CIFAR10(root='/expanse/lustre/projects/csd716/amirhesam/data', train=True,
                                            download=True)
    testset = torchvision.datasets.CIFAR10(root='/expanse/lustre/projects/csd716/amirhesam/data', train=False,
                                           download=True)

    knots_x, knots_y = np.array(trainset.data)/255.0,torch.tensor(trainset.targets)
    test_forEP2, testy =  np.array(testset.data)/255.0,torch.tensor(testset.targets)

    kernel_fn = lambda x, y: laplacian_split(x, y, bandwidth=20.0)

    result = vanilaep(
        kernel_fn, knots_x.astype('float32'), #.reshape(knots_x.shape[0],-1)
        one_hot(knots_y),
        test_forEP2.astype('float32'), #.reshape(test_forEP2.shape[0],-1)
        one_hot(testy),
        devices[0])
    epoch_keys = [i for i in result.keys()]
    acc_ep2_test = result[epoch_keys[-1]][1]['multiclass-acc']

