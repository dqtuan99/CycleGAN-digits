# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:14:26 2024

@author: Tuan
"""

from model import Discriminator, Generator
from data_loader import ImgDomainAdaptationData
from utils import generate_imgs
from torch import optim
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_cuda = torch.cuda.is_available()


class EarlyStopping:
    """Early stops the training if cycle consistency loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, epsilon=1.01):
        """
        Args:
            patience (int): How long to wait after last time cycle consistency loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.cycle_loss_min = np.Inf
        self.epsilon = epsilon

    def __call__(self, cycle_loss, gen, dis):

        score = -cycle_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(cycle_loss, gen, dis)
        elif score < self.best_score / self.epsilon:
            self.counter += 1
            print(f'\nCurrent cycle consistency loss {cycle_loss:.6f} > {self.cycle_loss_min:.6f}/{self.epsilon} = {self.cycle_loss_min/self.epsilon:.6f}')
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(cycle_loss, gen, dis)
            self.counter = 0

    def save_checkpoint(self, cycle_loss, gen, dis):
        '''Saves model when the cycle consistency loss decrease.'''

        gen_ab, gen_ba = gen
        dis_a, dis_b = dis

        if self.verbose:
            print(f'\ncycle consistency loss decreased ({self.cycle_loss_min:.6f} --> {cycle_loss:.6f}).  Saving model ...\n')
        # Note: Here you should define how you want to save your model. For example:
        torch.save(gen_ab.state_dict(), os.path.join(model_path, f'gen_{DS_NAME[dm_a]}_{DS_NAME[dm_b]}.pkl'))
        torch.save(gen_ba.state_dict(), os.path.join(model_path, f'gen_{DS_NAME[dm_b]}_{DS_NAME[dm_a]}.pkl'))

        torch.save(dis_a.state_dict(), os.path.join(model_path, f'dis_{DS_NAME[dm_a]}.pkl'))
        torch.save(dis_b.state_dict(), os.path.join(model_path, f'dis_{DS_NAME[dm_b]}.pkl'))

        self.cycle_loss_min = cycle_loss


MNIST_path = './dataset/MNIST_train.pt'
MNISTM_path = './dataset/MNISTM_train.pt'
SYN_path = './dataset/SYN_train.pt'
USPS_path = './dataset/USPS_train.pt'

ds_path = [MNIST_path, MNISTM_path, SYN_path, USPS_path]

MNIST_domain_id = 0
MNISTM_domain_id = 1
SYN_domain_id = 2
USPS_domain_id = 3

DS_NAME = ["MNIST", "MNISTM", "SYN", "USPS"]

EPOCHS = 100  # 50-300
BATCH_SIZE = 128
IMGS_TO_DISPLAY = 32

IMAGE_SIZE = 32
NUM_DOMAINS = 2

GRADIENT_PENALTY = 10

CONV_DIM = 12

model_path = './model'
samples_path = './samples'
os.makedirs(model_path, exist_ok=True)
os.makedirs(samples_path, exist_ok=True)

ab_combinations = np.array([[0,2], [0,3], [1,2], [1,3]])

for dm_a, dm_b in ab_combinations:

    # dm_a = 0
    # dm_b = 1

    model_path = './model'
    samples_path = './samples'

    # Define generators and discriminators
    gen_ab = Generator(in_channels=3, out_channels=3, conv_dim=CONV_DIM).to(device).train()
    gen_ba = Generator(in_channels=3, out_channels=3, conv_dim=CONV_DIM).to(device).train()

    dis_a = Discriminator(channels=3).to(device).train()
    dis_b = Discriminator(channels=3).to(device).train()

    # Define Optimizers
    g_optim = optim.Adam(list(gen_ab.parameters()) + list(gen_ba.parameters()),
                         lr=0.0001,
                         betas=(0.5, 0.999))

    d_optim = optim.Adam(list(dis_a.parameters()) + list(dis_b.parameters()),
                         lr=0.0001,
                         betas=(0.5, 0.999))

    # Data loaders
    data = ImgDomainAdaptationData(ds_path[dm_a], ds_path[dm_b], dm_a, dm_b, IMAGE_SIZE, IMAGE_SIZE)
    ds_loader = torch.utils.data.DataLoader(data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)
    iters_per_epoch = len(ds_loader)

    # Fix images for viz
    loader_iter = iter(ds_loader)
    img_fixed = next(loader_iter)
    a_fixed, b_fixed = img_fixed
    a_fixed = a_fixed[:IMGS_TO_DISPLAY].to(device)
    b_fixed = b_fixed[:IMGS_TO_DISPLAY].to(device)

    current_setting = f'{DS_NAME[dm_a]}_{DS_NAME[dm_b]}_conv{CONV_DIM}_batch{BATCH_SIZE}'

    print(f'Current train setting: {current_setting}')

    model_path = os.path.join(model_path, current_setting)
    os.makedirs(model_path, exist_ok=True)
    samples_path = os.path.join(samples_path, current_setting)
    samples_path_ab = os.path.join(samples_path, f'{DS_NAME[dm_a]}_to_{DS_NAME[dm_b]}')
    samples_path_ba = os.path.join(samples_path, f'{DS_NAME[dm_b]}_to_{DS_NAME[dm_a]}')
    os.makedirs(samples_path, exist_ok=True)
    os.makedirs(samples_path_ab, exist_ok=True)
    os.makedirs(samples_path_ba, exist_ok=True)

    early_stopping = EarlyStopping(patience=12, verbose=True)

    g_adv_loss_per_ep = []
    g_clf_loss_per_ep = []
    g_loss_per_ep = []

    d_loss_per_ep = []

    train_info = []

    # CycleGan Training
    for epoch in range(EPOCHS):

        g_adv_losses = []
        g_cyc_losses = []
        g_losses = []

        d_losses = []

        for batch_idx, batch_data in tqdm(enumerate(ds_loader), total=iters_per_epoch, desc=f'Epoch {epoch+1}'):
            a_real, b_real = batch_data
            a_real, b_real = a_real.to(device), b_real.to(device)

            # Fake Images
            b_fake = gen_ab(a_real)
            a_fake = gen_ba(b_real)

            # Train discriminators
            a_real_out = dis_a(a_real)
            a_fake_out = dis_a(a_fake.detach())
            d_a_loss = (torch.mean((a_real_out - 1) ** 2) + torch.mean(a_fake_out ** 2)) / 2

            b_real_out = dis_b(b_real)
            b_fake_out = dis_b(b_fake.detach())
            d_b_loss = (torch.mean((b_real_out - 1) ** 2) + torch.mean(b_fake_out ** 2)) / 2

            d_optim.zero_grad()
            d_loss = d_a_loss + d_b_loss
            d_loss.backward()
            d_optim.step()

            # Training generators
            a_fake_out = dis_a(a_fake)
            b_fake_out = dis_b(b_fake)

            g_a_adv_loss = torch.mean((a_fake_out - 1) ** 2)
            g_b_adv_loss = torch.mean((b_fake_out - 1) ** 2)
            g_adv_loss = g_a_adv_loss + g_b_adv_loss

            g_a_cyc_loss = (a_real - gen_ba(b_fake)).abs().mean()
            g_b_cyc_loss = (b_real - gen_ab(a_fake)).abs().mean()
            g_cyc_loss = g_a_cyc_loss + g_b_cyc_loss

            g_optim.zero_grad()
            g_loss = g_adv_loss + 10 * g_cyc_loss
            g_loss.backward()
            g_optim.step()

            g_adv_losses.append(g_adv_loss.item())
            g_cyc_losses.append(g_cyc_loss.item())
            g_losses.append(g_loss.item())

            d_losses.append(d_loss.item())

        generate_imgs(a_fixed, b_fixed, gen_ab, gen_ba, (samples_path_ab, samples_path_ba), DS_NAME[dm_a], DS_NAME[dm_b], epoch+1)

        avg_g_adv_loss = np.mean(g_adv_losses)
        avg_g_cyc_loss = np.mean(g_cyc_losses)
        avg_g_loss = np.mean(g_losses)

        avg_d_loss = np.mean(d_losses)

        train_info.append([avg_g_adv_loss, avg_g_cyc_loss, avg_g_loss, avg_d_loss])

        print(f'\nGenerator:\nadv_loss: {avg_g_adv_loss:.6f}, cyc_loss: {avg_g_cyc_loss:.6f}')
        print(f'Total loss: {avg_g_loss:.6f}')
        print("------------------------------")
        print('Discriminator:')
        print(f'Total loss: {avg_d_loss:.6f}')
        print("==============================")

        early_stopping(avg_g_cyc_loss, (gen_ab, gen_ba), (dis_a, dis_b))
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    generate_imgs(a_fixed, b_fixed, gen_ab, gen_ba, (samples_path_ab, samples_path_ba), DS_NAME[dm_a], DS_NAME[dm_b], -1)

    df = pd.DataFrame(train_info, columns=['Gen Adv Loss', 'Gen Cyc Loss',
                                           'Gen Total Loss',
                                           'Dis Total Loss'])

    train_info_path = os.path.join('.', 'train_info', current_setting)
    os.makedirs(train_info_path, exist_ok=True)
    train_info_path = os.path.join(train_info_path, 'train_info.csv')
    df.to_csv(train_info_path, index=True)

    print(f'Saving train info to {train_info_path}')

print('All done')