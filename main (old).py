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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

EPOCHS = 100  # 50-300
BATCH_SIZE = 128
LOAD_MODEL = False

IMAGE_SIZE = 32
NUM_DOMAINS = 4

DS_NAME = ["MNIST", "MNISTM", "SVHN", "SynDigits"]    


# Directories for storing model and output samples
model_path = './model/' + current_time
os.makedirs(model_path, exist_ok=True)
samples_path = './samples/' + current_time
os.makedirs(samples_path, exist_ok=True)
db_path = './data/' + current_time
os.makedirs(samples_path, exist_ok=True)

# Networks

# ab_gen = Generator(in_channels=A_Channels, out_channels=B_Channels)
# ba_gen = Generator(in_channels=B_Channels, out_channels=A_Channels)
# a_disc = Discriminator(channels=A_Channels)
# b_disc = Discriminator(channels=B_Channels)

# # Load previous model

# if LOAD_MODEL:
#     ab_gen.load_state_dict(torch.load(os.path.join(model_path, 'ab_gen.pkl')))
#     ba_gen.load_state_dict(torch.load(os.path.join(model_path, 'ba_gen.pkl')))
#     a_disc.load_state_dict(torch.load(os.path.join(model_path, 'a_disc.pkl')))
#     b_disc.load_state_dict(torch.load(os.path.join(model_path, 'b_disc.pkl')))

generators = [x[:] for x in [[None] * NUM_DOMAINS] * NUM_DOMAINS]
discriminators = [None] * NUM_DOMAINS

g_optim = SymmetricMatrix(NUM_DOMAINS)
d_optim = SymmetricMatrix(NUM_DOMAINS)

for a in range(NUM_DOMAINS):
    for b in range(NUM_DOMAINS):
        if a == b:
            continue
        generators[a][b] = Generator(in_channels=3, out_channels=3).to(device)

for a in range(NUM_DOMAINS):        
    discriminators[a] = Discriminator(channels=3).to(device)


# Define Optimizers
# g_opt = optim.Adam(list(ab_gen.parameters()) + list(ba_gen.parameters()), lr=0.0002, betas=(0.5, 0.999),
#                    weight_decay=2e-5)
# d_opt = optim.Adam(list(a_disc.parameters()) + list(b_disc.parameters()), lr=0.0002, betas=(0.5, 0.999),
#                    weight_decay=2e-5)    

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

# Data loaders
# a_loader, b_loader = get_loaders(db_path, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, a_ds=A_DS, b_ds=B_DS)
# iters_per_epoch = min(len(a_loader), len(b_loader))

data = ImgDomainAdaptationData(IMAGE_SIZE, IMAGE_SIZE)
ds_loader = torch.utils.data.DataLoader(dataset=data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        drop_last=True)

iters_per_epoch = len(ds_loader)

# Fix images for viz
ds_iter = iter(ds_loader)
fixed = next(ds_iter)

real = [x[:] for x in [[None] * NUM_DOMAINS] * NUM_DOMAINS]

# Cycle-GAN Training
for epoch in range(EPOCHS):
    for a in range(NUM_DOMAINS):
        for b in range(NUM_DOMAINS):
            if a == b:
                continue
            generators[a][b].train()
        discriminators[a].train()
    
    train_bar = tqdm(ds_loader, desc="Epoch {}'s progress bar: ".format(epoch+1))
    
    for step, data in enumerate(train_bar):
        
        # Loading data
        real = data
        
        for i in range(NUM_DOMAINS):
            real[i] = real[i].to(device)

        for a in range(NUM_DOMAINS):
            for b in range(NUM_DOMAINS):
                if a == b:
                    continue
                
                # Fake Images
                b_fake = generators[a][b](real[a])
                a_fake = generators[b][a](real[b])
                
                # Training Discriminator
                a_real_out = discriminators[a](real[a])
                a_fake_out = discriminators[a](a_fake.detach())
                a_d_loss = (torch.mean((a_real_out - 1) ** 2) + torch.mean(a_fake_out ** 2)) / 2
                
                b_real_out = discriminators[b](real[b])
                b_fake_out = discriminators[b](b_fake.detach())
                b_d_loss = (torch.mean((b_real_out - 1) ** 2) + torch.mean(b_fake_out ** 2)) / 2
                
                d_optim[a, b].zero_grad()
                d_loss = a_d_loss + b_d_loss                    
                d_loss.backward()
                d_optim[a, b].step()
                
                # Training Generator
                a_fake_out = discriminators[a](a_fake)
                b_fake_out = discriminators[b](b_fake)
                
                a_g_loss = torch.mean((a_fake_out - 1) ** 2)
                b_g_loss = torch.mean((b_fake_out - 1) ** 2)
                g_gan_loss = a_g_loss + b_g_loss
                
                a_g_ctnt_loss = (real[a] - generators[a][b](b_fake)).abs().mean()
                b_g_ctnt_loss = (real[b] - generators[b][a](a_fake)).abs().mean()
                g_ctnt_loss = a_g_ctnt_loss + b_g_ctnt_loss
                
                g_optim[a, b].zero_grad()
                g_loss = g_gan_loss + g_ctnt_loss
                g_loss.backward()
                g_optim[a, b].step()
                
                if step % 50 == 0:
                    print("\nPerform training on domain (a, b) = ({}, {}) pair".format(a,b))
                    print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
                          + " it: " + str(step) + "/" + str(iters_per_epoch)
                          + "\ta_d_loss:" + str(round(a_d_loss.item(), 4))
                          + "\ta_g_loss:" + str(round(a_g_loss.item(), 4))
                          + "\ta_g_ctnt_loss:" + str(round(a_g_ctnt_loss.item(), 4))
                          + "\tb_d_loss:" + str(round(b_d_loss.item(), 4))
                          + "\tb_g_loss:" + str(round(b_g_loss.item(), 4))
                          + "\tb_g_ctnt_loss:" + str(round(b_g_ctnt_loss.item(), 4)))
                
    for a in range(NUM_DOMAINS):
        for b in range(NUM_DOMAINS):
            if a == b:
                continue
            fname = os.path.join(model_path, '{}_{}_generator.pkl'.format(DS_NAME[a], DS_NAME[b]))
            torch.save(generators[a][b].state_dict(), fname)
            fname = os.path.join(model_path, '{}_{}_generator.pkl'.format(DS_NAME[b], DS_NAME[a]))
            torch.save(generators[b][a].state_dict(), fname)
            generate_imgs(fixed[a], fixed[b], 
                          generators[a][b], generators[b][a], 
                          samples_path, 
                          DS_NAME[a], DS_NAME[b], 
                          epoch=epoch + 1)
            
        fname = os.path.join(model_path, '{}_discriminator.pkl'.format(DS_NAME[a]))
        torch.save(discriminators[a].state_dict(), fname)
        
for a in range(NUM_DOMAINS):
    for b in range(NUM_DOMAINS):
        if a == b:
            continue        
        generate_imgs(fixed[a], fixed[b], 
                      generators[a][b], generators[b][a], 
                      samples_path, 
                      DS_NAME[a], DS_NAME[b])

        # # Fake Images
        # b_fake = ab_gen(a_real)
        # a_fake = ba_gen(b_real)

        # # Training discriminator
        # a_real_out = a_disc(a_real)
        # a_fake_out = a_disc(a_fake.detach())
        # a_d_loss = (torch.mean((a_real_out - 1) ** 2) + torch.mean(a_fake_out ** 2)) / 2

        # b_real_out = b_disc(b_real)
        # b_fake_out = b_disc(b_fake.detach())
        # b_d_loss = (torch.mean((b_real_out - 1) ** 2) + torch.mean(b_fake_out ** 2)) / 2

        # d_opt.zero_grad()
        # d_loss = a_d_loss + b_d_loss
        # d_loss.backward()
        # d_opt.step()

        # # Training Generator
        # a_fake_out = a_disc(a_fake)
        # b_fake_out = b_disc(b_fake)

        # a_g_loss = torch.mean((a_fake_out - 1) ** 2)
        # b_g_loss = torch.mean((b_fake_out - 1) ** 2)
        # g_gan_loss = a_g_loss + b_g_loss

        # a_g_ctnt_loss = (a_real - ba_gen(b_fake)).abs().mean()
        # b_g_ctnt_loss = (b_real - ab_gen(a_fake)).abs().mean()
        # g_ctnt_loss = a_g_ctnt_loss + b_g_ctnt_loss

        # g_opt.zero_grad()
        # g_loss = g_gan_loss + g_ctnt_loss
        # g_loss.backward()
        # g_opt.step()

        # if step % 50 == 0:
        #     print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
        #           + " it: " + str(step) + "/" + str(iters_per_epoch)
        #           + "\ta_d_loss:" + str(round(a_d_loss.item(), 4))
        #           + "\ta_g_loss:" + str(round(a_g_loss.item(), 4))
        #           + "\ta_g_ctnt_loss:" + str(round(a_g_ctnt_loss.item(), 4))
        #           + "\tb_d_loss:" + str(round(b_d_loss.item(), 4))
        #           + "\tb_g_loss:" + str(round(b_g_loss.item(), 4))
        #           + "\tb_g_ctnt_loss:" + str(round(b_g_ctnt_loss.item(), 4)))

    # torch.save(ab_gen.state_dict(), os.path.join(model_path, 'ab_gen.pkl'))
    # torch.save(ba_gen.state_dict(), os.path.join(model_path, 'ba_gen.pkl'))
    # torch.save(a_disc.state_dict(), os.path.join(model_path, 'a_disc.pkl'))
    # torch.save(b_disc.state_dict(), os.path.join(model_path, 'b_disc.pkl'))

#     generate_imgs(a_fixed, b_fixed, ab_gen, ba_gen, samples_path, epoch=epoch + 1)

# generate_imgs(a_fixed, b_fixed, ab_gen, ba_gen, samples_path)
