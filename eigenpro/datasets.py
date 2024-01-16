import numpy as np

import torch
# import torchvision
# import torchvision.transforms as transforms
from torch.nn.functional import one_hot
import os
from os.path import join as pjoin

import ipdb


class Cifar5mDataset():

    def __init__(self,
                 DATADIR='/expanse/lustre/projects/csd716/parthepandit/data/cifar-5m/',
                 parts=4,
                 device=torch.device('cpu'), subsample =None,
                 n_test=100_000,num_knots= None,
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        print('Loading cifar5m train set...')
        for ind in range(parts+1):
            print(f'part={ind}')
            # z = np.load(pjoin(DATADIR, f'part{i}.npz'))
            self.X_train.append( torch.load(pjoin(DATADIR,f'part{ind}_X.pt'), torch.device('cpu')) )
            self.y_train.append(torch.load(pjoin(DATADIR,f'part{ind}_y.pt'), torch.device('cpu')))
            # print(f'Loaded part {i + 1}/6')
        print("Loading cifar5m test set...")
        # z = np.load(pjoin(DATADIR, 'part5.npz'))
        self.X_test.append(torch.load(pjoin(DATADIR,f'part5_X.pt'), torch.device('cpu'))[:n_test])
        self.y_test.append(torch.load(pjoin(DATADIR,f'part5_y.pt'), torch.device('cpu'))[:n_test])

        self.X_train = torch.cat(self.X_train)
        self.y_train = torch.cat(self.y_train)
        self.X_test = torch.cat(self.X_test)
        self.y_test = torch.cat(self.y_test)

        # address = "/expanse/lustre/scratch/amirhesam/temp_project/cifar5m/"
        # n = 2_000_000
        # ids = np.random.choice(self.X_train.shape[0],n)
        # X_train_save = self.X_train[ids,:]
        # Y_train_save = self.y_train[ids]
        # torch.save(X_train_save,address+'X_train_2M')
        # torch.save(Y_train_save, address + 'Y_train_2M')
        #
        # n = 10_000
        # ids = np.random.choice(self.X_test.shape[0],n)
        # X_test_save = self.X_test[ids,:]
        # Y_test_save = self.y_test[ids]
        # torch.save(X_test_save,address+'X_test_10K')
        # torch.save(Y_test_save, address + 'Y_test_10K')



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