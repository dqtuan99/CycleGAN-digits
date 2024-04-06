# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:49:04 2024

@author: Tuan
"""

import torch
from torchvision import transforms
import numpy as np


# MNIST_path = './dataset/MNIST_train.pt'
# MNISTM_path = './dataset/MNISTM_train.pt'
# SVHN_path = './dataset/SVHN_train.pt'
# SYN_path = './dataset/SYN_train.pt'
# USPS_path = './dataset/USPS_train.pt'

# ds_path = [MNIST_path, MNISTM_path, SVHN_path, SYN_path, USPS_path]

# MNIST_domain_id = 0
# MNISTM_domain_id = 1
# SVHN_domain_id = 2
# SYN_domain_id = 3
# USPS_domain_id = 4

MNIST_path = './dataset/MNIST_train.pt'
MNISTM_path = './dataset/MNISTM_train.pt'
SYN_path = './dataset/SYN_train.pt'
USPS_path = './dataset/USPS_train.pt'

ds_path = [MNIST_path, MNISTM_path, SYN_path, USPS_path]

MNIST_domain_id = 0
MNISTM_domain_id = 1
SYN_domain_id = 2
USPS_domain_id = 3


class ImgDomainAdaptationData(torch.utils.data.Dataset):
    def __init__(self, path_A, path_B, id_A, id_B, w, h):

        self.transform = transforms.Compose([transforms.Resize([w, h]),
                                            transforms.Normalize([0.5], [0.5])])

        self.data_A = torch.load(path_A)
        self.data_B = torch.load(path_B)

        self.img_A = self.transform(self.data_A[0])
        self.img_B = self.transform(self.data_B[0])

        self.label_A = self.data_A[1]
        self.label_B = self.data_B[1]

        self.img_A, self.domain_A = self.pre_processing(self.img_A, id_A)
        self.img_B, self.domain_B = self.pre_processing(self.img_B, id_B)

        self.len_A = self.label_A.shape[0]
        self.len_B = self.label_B.shape[0]

        # self.len = min(self.len_A, self.len_B)


    def pre_processing(self, img, domain):
        num_img = img.shape[0]

        if len(img.shape) < 4:
            img = img.unsqueeze(1).repeat(1, 3, 1, 1)

        domain_label = np.zeros(num_img, dtype=int) + domain

        return img, domain_label

    def __len__(self):
        # return self.len
        return 60000

    def __getitem__(self, index):
        index_A = index % self.len_A
        index_B = index % self.len_B

        return self.img_A[index_A], self.img_B[index_B]