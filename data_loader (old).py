import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os


MNIST_path = './dataset/MNIST_train.pt'
MNISTM_path = './dataset/MNISTM_train.pt'
# SVHN_path = './dataset/SVHN_train.pt'
SYN_path = './dataset/SYN_train.pt'
USPS_path = './dataset/USPS_train.pt'


class ImgDomainAdaptationData(torch.utils.data.Dataset):
    def __init__(self, w, h):
        
        self.transform = transforms.Compose([transforms.Resize([w, h]),
                                            transforms.Normalize([0.5], [0.5])])
            
        self.MNIST_data = torch.load(MNIST_path)
        self.MNISTM_data = torch.load(MNISTM_path)
        # self.SVHN_data = torch.load(SVHN_path)
        self.SYN_data = torch.load(SYN_path)
        self.USPS_data = torch.load(USPS_path)
        
        self.MNIST_img = self.transform(self.MNIST_data[0])
        self.MNISTM_img = self.transform(self.MNISTM_data[0])
        # self.SVHN_img = self.transform(self.SVHN_data[0])
        self.SYN_img = self.transform(self.SYN_data[0])
        self.USPS_img = self.transform(self.USPS_data[0])
        
        self.MNIST_label = self.MNIST_data[1]
        self.MNISTM_label = self.MNISTM_data[1]
        # self.SVHN_label = self.SVHN_data[1]
        self.SYN_label = self.SYN_data[1]
        self.USPS_label = self.USPS_data[1]
        
        self.MNIST_img, self.MNIST_domain = self.pre_processing(self.MNIST_img, 0)
        self.MNISTM_img, self.MNISTM_domain = self.pre_processing(self.MNISTM_img, 1)
        # self.SVHN_img, self.SVHN_domain = self.pre_processing(self.SVHN_img, 2)
        self.SYN_img, self.SYN_domain = self.pre_processing(self.SYN_img, 2)
        self.USPS_img, self.USPS_domain = self.pre_processing(self.USPS_img, 3)
        
        # self.img = [self.MNIST_img, self.MNISTM_img, self.SVHN_img, self.SYN_img]
        
    
    def pre_processing(self, img, domain):
        num_img = img.shape[0]
        
        if len(img.shape) < 4:
            img = img.unsqueeze(1).repeat(1, 3, 1, 1)
        
        domain_label = np.zeros(num_img, dtype=int) + domain
                
        return img, domain_label
        
    def __len__(self):
        length = np.array([self.MNIST_label.shape[0],
                           self.MNISTM_label.shape[0],
                           self.SVHN_label.shape[0],
                           self.SYN_label.shape[0]])
        length = np.min(length)
        return length
    
    def __getitem__(self, index):
        return self.MNIST_img[index], self.MNISTM_img[index], \
            self.SVHN_img[index], self.SYN_img[index]

# import torch
# from torchvision import datasets
# from torchvision import transforms
# import numpy as np
# import os


# def get_dataset(ds, ds_path, transform):
#     ds_full_path = os.path.join(ds_path, ds)
#     if ds.lower() == 'svhn':
#         return datasets.SVHN(ds_full_path, split='train', download=True, transform=transform)
#     elif ds.lower() == 'mnist':
#         return datasets.MNIST(ds_full_path, train=True, download=True, transform=transform)
#     elif ds.lower() == 'usps':
#         return datasets.USPS(ds_full_path, train=True, download=True, transform=transform)
#     else:
#         return datasets.ImageFolder(ds_full_path, transform=transform)


# def get_loaders(ds_path='./data', batch_size=128, image_size=32, a_ds='svhn', b_ds='mnist'):
#     mean = np.array([0.5])
#     std = np.array([0.5])

#     transform = transforms.Compose([transforms.Resize([image_size, image_size]),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean, std)])

#     a_ds = get_dataset(a_ds, ds_path, transform=transform)
#     b_ds = get_dataset(b_ds, ds_path, transform=transform)

#     a_ds_loader = torch.utils.data.DataLoader(dataset=a_ds,
#                                               batch_size=batch_size,
#                                               shuffle=True,
#                                               num_workers=8,
#                                               drop_last=True)

#     b_ds_loader = torch.utils.data.DataLoader(dataset=b_ds,
#                                               batch_size=batch_size,
#                                               shuffle=True,
#                                               num_workers=8,
#                                               drop_last=True)
#     return a_ds_loader, b_ds_loader

