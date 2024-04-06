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
import datetime

class SymmetricMatrix:
    def __init__(self, size):
        self.size = size
        self.matrix = [x[:] for x in [[None] * size] * size]

    def __getitem__(self, index):
        a, b = index
        if a > b:
            a, b = b, a
        return self.matrix[a][b]

    def __setitem__ (self, index, obj):
        a, b = index
        self.matrix[a][b] = obj




EPOCHS = 300  # 50-300
BATCH_SIZE = 64
IMGS_TO_DISPLAY = 25

IMAGE_SIZE = 32
NUM_DOMAINS = 4

N_CRITIC = 5
GRADIENT_PENALTY = 10

CONV_DIM = 12

model_path = './model'
os.makedirs(model_path, exist_ok=True)
samples_path = './samples'
os.makedirs(samples_path, exist_ok=True)

DS_NAME = ["MNIST", "MNISTM", "SYN", "USPS"]

# Define generators and discriminators
generators = [x[:] for x in [[None] * NUM_DOMAINS] * NUM_DOMAINS]
discriminators = [None] * NUM_DOMAINS

g_optim = SymmetricMatrix(NUM_DOMAINS)
d_optim = SymmetricMatrix(NUM_DOMAINS)

for a in range(NUM_DOMAINS):
  for b in range(NUM_DOMAINS):
    if a == b:
      continue
    generators[a][b] = Generator(in_channels=3, out_channels=3, conv_dim=CONV_DIM).to(device)
    generators[a][b].train()

for a in range(NUM_DOMAINS):
    discriminators[a] = Discriminator(channels=3).to(device)
    discriminators[a].train()

# Define Optimizers
for a in range(NUM_DOMAINS - 1):
    for b in range(a + 1, NUM_DOMAINS):
        g_optim[a, b] = optim.Adam(list(generators[a][b].parameters()) + \
                                   list(generators[b][a].parameters()),
                                   lr=0.0002,
                                   betas=(0.5, 0.999),
                                   weight_decay=2e-5)

        d_optim[a, b] = optim.Adam(list(discriminators[a].parameters()) + \
                                   list(discriminators[b].parameters()),
                                   lr=0.0002,
                                   betas=(0.5, 0.999),
                                   weight_decay=2e-5)

for a in range(NUM_DOMAINS):
    for b in range(NUM_DOMAINS):
        if a == b:
            continue

        # Data loaders
        data = ImgDomainAdaptationData(ds_path[a], ds_path[b], a, b, 32, 32)
        ds_loader = torch.utils.data.DataLoader(data,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)
        iters_per_epoch = len(ds_loader)

        # Fix images for viz
        loader_iter = iter(ds_loader)
        img_fixed = next(loader_iter)

        # Train

        print("\nPerform training on domain (a, b) = ({}, {}) pair".format(a,b))

        for epoch in range(EPOCHS):
            for step, data in enumerate(ds_loader):
                real = data
                for i in range(2):
                  real[i] = real[i].to(device)

                # Fake img
                b_fake = generators[a][b](real[0])
                a_fake = generators[b][a](real[1])

                # Training Discriminators
                a_real_out = discriminators[a](real[0])
                a_fake_out = discriminators[a](a_fake.detach())
                a_d_loss = (torch.mean((a_real_out - 1) ** 2) + torch.mean(a_fake_out ** 2)) / 2

                b_real_out = discriminators[b](real[1])
                b_fake_out = discriminators[b](b_fake.detach())
                b_d_loss = (torch.mean((b_real_out - 1) ** 2) + torch.mean(b_fake_out ** 2)) / 2

                d_optim[a, b].zero_grad()
                d_loss = a_d_loss + b_d_loss
                d_loss.backward()
                d_optim[a, b].step()

                # Training Generators
                a_fake_out = discriminators[a](a_fake)
                b_fake_out = discriminators[b](b_fake)

                a_g_loss = torch.mean((a_fake_out - 1) ** 2)
                b_g_loss = torch.mean((b_fake_out - 1) ** 2)
                g_gan_loss = a_g_loss + b_g_loss

                a_g_ctnt_loss = (real[0] - generators[a][b](b_fake)).abs().mean()
                b_g_ctnt_loss = (real[1] - generators[b][a](a_fake)).abs().mean()
                g_ctnt_loss = a_g_ctnt_loss + b_g_ctnt_loss

                g_optim[a, b].zero_grad()
                g_loss = g_gan_loss + g_ctnt_loss
                g_loss.backward()
                g_optim[a, b].step()

                if step % 50 == 0:
                    print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
                          + " it: " + str(step) + "/" + str(iters_per_epoch)
                          + "\ta_d_loss:" + str(round(a_d_loss.item(), 4))
                          + "\ta_g_loss:" + str(round(a_g_loss.item(), 4))
                          + "\ta_g_ctnt_loss:" + str(round(a_g_ctnt_loss.item(), 4))
                          + "\tb_d_loss:" + str(round(b_d_loss.item(), 4))
                          + "\tb_g_loss:" + str(round(b_g_loss.item(), 4))
                          + "\tb_g_ctnt_loss:" + str(round(b_g_ctnt_loss.item(), 4)))

            fname = os.path.join(model_path, '{}_{}_generator.pkl'.format(DS_NAME[a], DS_NAME[b]))
            torch.save(generators[a][b].state_dict(), fname)
            fname = os.path.join(model_path, '{}_{}_generator.pkl'.format(DS_NAME[b], DS_NAME[a]))
            torch.save(generators[b][a].state_dict(), fname)

            fname = os.path.join(model_path, '{}_discriminator.pkl'.format(DS_NAME[a]))
            torch.save(discriminators[a].state_dict(), fname)