import numpy as np

import torch
from torch.nn.functional import one_hot
import os
from os.path import join as pjoin


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
            self.X_train.append( torch.load(pjoin(DATADIR,f'part{ind}_X.pt'), torch.device('cpu')) )
            self.y_train.append(torch.load(pjoin(DATADIR,f'part{ind}_y.pt'), torch.device('cpu')))
        print("Loading cifar5m test set...")
        self.X_test.append(torch.load(pjoin(DATADIR,f'part5_X.pt'), torch.device('cpu'))[:n_test])
        self.y_test.append(torch.load(pjoin(DATADIR,f'part5_y.pt'), torch.device('cpu'))[:n_test])

        self.X_train = torch.cat(self.X_train)
        self.y_train = torch.cat(self.y_train)
        self.X_test = torch.cat(self.X_test)
        self.y_test = torch.cat(self.y_test)

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
            self.X_train.append( torch.load(pjoin(DATADIR,f'ciar5m_mobilenetv2_100_feature_train_{ind}.pt'), torch.device('cpu')) )
            self.y_train.append(torch.load(pjoin(DATADIR,f'ciar5m_mobilenetv2_100_y_train_{ind}.pt'), torch.device('cpu')))
        print("Loading cifar5M_mobilenet test set...")
        self.X_test.append(torch.load(pjoin(DATADIR,f'ciar5m_mobilenetv2_100_feature_test.pt'), torch.device('cpu'))[:n_test])
        self.y_test.append(torch.load(pjoin(DATADIR,f'ciar5m_mobilenetv2_100_y_test.pt'), torch.device('cpu'))[:n_test])

        self.X_train = torch.cat(self.X_train)
        self.y_train = torch.cat(self.y_train)
        self.X_test = torch.cat(self.X_test)
        self.y_test = torch.cat(self.y_test)

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
