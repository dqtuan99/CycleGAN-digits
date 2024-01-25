import torch
import torchvision.utils as vutils
import math
import os


def generate_imgs(a, b, ab_gen, ba_gen, samples_path, a_name, b_name, epoch=0):
    ab_gen.eval()
    ba_gen.eval()

    b_fake = ab_gen(a)
    a_fake = ba_gen(b)

    a_imgs = torch.zeros((a.shape[0] * 2, 3, a.shape[2], a.shape[3]))
    b_imgs = torch.zeros((b.shape[0] * 2, 3, b.shape[2], b.shape[3]))

    even_idx = torch.arange(start=0, end=a.shape[0] * 2, step=2)
    odd_idx = torch.arange(start=1, end=a.shape[0] * 2, step=2)

    a_imgs[even_idx] = a.cpu()
    a_imgs[odd_idx] = b_fake.cpu()

    b_imgs[even_idx] = b.cpu()
    b_imgs[odd_idx] = a_fake.cpu()

    rows = math.ceil((a.shape[0] * 2) ** 0.5)
    a_imgs_ = vutils.make_grid(a_imgs, normalize=True, nrow=rows)
    b_imgs_ = vutils.make_grid(b_imgs, normalize=True, nrow=rows)

    vutils.save_image(a_imgs_, os.path.join(samples_path, '{}_to_{}_{}.png'.format(a_name, b_name, epoch)))
    vutils.save_image(b_imgs_, os.path.join(samples_path, '{}_to_{}_{}.png'.format(b_name, a_name, epoch)))
