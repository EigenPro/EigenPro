import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
# import torchvision
# import torchvision.transforms as transforms
from torch.nn.functional import one_hot
import os
from os.path import join as pjoin

import ipdb


class cifar5m(Dataset):

    def __init__(self,
                 DATADIR='/expanse/lustre/projects/csd716/parthepandit/data/cifar-5m/',
                 parts=1,
                 device=torch.device('cpu'), subsample=None,
                 n_test=100000, num_knots=None, knot_include=0,
                 **kwargs):
        super().__init__(**kwargs)

        test_dir = '/expanse/lustre/projects/csd716/amirhesam/data/cifar5m_raw_test_subset/'
        self.X_train = []
        self.y_train = []
        print('Loading cifar5m train set...')
        for ind in range(parts+1):
            print(f'part={ind}')
            self.X_train.append( torch.load(pjoin(DATADIR,f'part{ind}_X.pt'), torch.device('cpu')) )
            self.y_train.append(torch.load(pjoin(DATADIR,f'part{ind}_y.pt'), torch.device('cpu')))
            # print(f'Loaded part {i + 1}/6')
        print("Loading cifar5m test set...")


        # self.X_test = torch.load(pjoin(DATADIR,f'part5_X.pt')).float()
        # self.y_test = torch.load(pjoin(DATADIR,f'part5_y.pt'))


        self.X_test = torch.load(pjoin(test_dir,f'x_test_20k'))
        self.y_test = torch.load(pjoin(test_dir,f'y_test_20k'))

        self.X_train = torch.cat(self.X_train).float()
        self.y_train = torch.cat(self.y_train)

        mean = self.X_train.mean(dim=0)
        std = self.X_train.std(dim=0)

        self.X_train = (self.X_train - mean)/std


class Cifar5M_mobilenet_Dataset():
    def __init__(self,
                 DATADIR='/taiga/nsf/delta/bbjr/abedsol1/dataset/cifar5m_mobilenet',
                 parts=1,
                 device=torch.device('cpu'), subsample =None,
                 n_test=100_000,num_knots= None,
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        print('Loading cifar5M_mobilenet train set...')
        for ind in range(parts+1):
            print(f'part={ind}')
            # z = np.load(pjoin(DATADIR, f'part{i}.npz'))
            self.X_train.append( torch.load(pjoin(DATADIR,f'ciar5m_mobilenetv2_100_feature_train_{ind}.pt'), torch.device('cpu')) )
            self.y_train.append(torch.load(pjoin(DATADIR,f'ciar5m_mobilenetv2_100_y_train_{ind}.pt'), torch.device('cpu')))
            # print(f'Loaded part {i + 1}/6')
        print("Loading cifar5M_mobilenet test set...")
        # z = np.load(pjoin(DATADIR, 'part5.npz'))
        self.X_test.append(torch.load(pjoin(DATADIR,f'ciar5m_mobilenetv2_100_feature_test.pt'), torch.device('cpu'))[:n_test])
        self.y_test.append(torch.load(pjoin(DATADIR,f'ciar5m_mobilenetv2_100_y_test.pt'), torch.device('cpu'))[:n_test])

        self.X_train = torch.cat(self.X_train)
        self.y_train = torch.cat(self.y_train)
        self.X_test = torch.cat(self.X_test)
        self.y_test = torch.cat(self.y_test)


        # address = "/scratch/bbjr/abedsol1/cifar5m_mobilenet"
        # n = 2_000_000
        # ids = np.random.choice(self.X_train.shape[0],n)
        # X_train_save = self.X_train[ids,:]
        # Y_train_save = self.y_train[ids]
        # torch.save(X_train_save,address+'X_train_2M')
        # torch.save(Y_train_save, address +'Y_train_2M')
        #
        # n = 10_000
        # ids = np.random.choice(self.X_test.shape[0],n)
        # X_test_save = self.X_test[ids,:]
        # Y_test_save = self.y_test[ids]
        # torch.save(X_test_save,address+'X_test_10K')
        # torch.save(Y_test_save, address +'Y_test_10K')
        #
        # ipdb.set_trace()



        # if num_knots is not None:
        #     randomind_knots = np.random.choice(
        #         range(self.y_train.shape[0]), size=num_knots, replace=False)
        #     self.knots_x = self.X_train[randomind_knots]
        #     self.knots_y = self.y_train[randomind_knots]
            # self.knots_x = torch.from_numpy(np.array(self.knots_x)).to(self.device)
            # self.knots_y = torch.from_numpy(np.array(self.knots_y)).to(self.device)

        if subsample is not None:
            if num_knots==None:
                diff_set = set(range(self.y_train.shape[0]))
            else:
                diff_set = set(range(self.y_train.shape[0])) - set(randomind_knots)
            diff_set = np.array(list(diff_set))
            randomind = np.random.choice(
                diff_set, size=subsample, replace=False)
            self.X_train = self.X_train[randomind]
            self.y_train = self.y_train[randomind]

        # ipdb.set_trace()