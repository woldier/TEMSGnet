"""
This class is coding
1. some basic network and tool functions related to the network
2. position encoding
3. res net

"""
import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# from einops import rearrange
# from datasets import load_dataset
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.nn as nn
from utils import *

"""
 ======================Hyperparameters of diffusion processes==================================
 
 The forward diffusion process gradually adds noise to an image from the real distribution,in a number of time steps TT. 
 This happens according to a variance schedule. The original DDPM authors employed a linear schedule:
    We set the forward process variances to constants increasing linearly from 1e-4 -> 2e-2
 However, it was shown in (Nichol et al., 2021) that better results can be achieved when employing a cosine schedule
"""


def cosine_beta_schedule(timesteps, s=0.008):
    """

    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    cosin β

    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """

    :param timesteps:
    :param beta_start:
    :param beta_end:
    :return:
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """

    :param timesteps:
    :param beta_start:
    :param beta_end:
    :return:
    """

    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


"""
==========================================================================
"""


def extract(a, t, x_shape):
    """
    The generated data is a one-dimensional array, but our input data is multi-dimensional,
    so we need to extend the one-dimensional array to be the same as the x_shape
    :param a:
    :param t:
    :param x_shape:
    :return:
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


"""
==========================forward=====================================
"""

timesteps = 200

betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def q_sample_reverse(x_end, t, noise=None):
    """
    Reverse the noisy data based on t and the noise at the time of noise addition
    :param x_end:
    :param t:
    :param noise:
    :return:
    """
    if noise is None:
        raise RuntimeError('noise must be not null')
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_end.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_end.shape
    )
    return (x_end - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t


"""
 =================loss_fn=================
"""


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1", condition=None):
    if noise is None:  # 如果自己没有给定噪声 则用标准的高斯白噪声
        noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)  # 从x0得到xt
    if condition is None:
        predicted_noise = denoise_model(x_noisy, t)  # 送入反向网络计算得到xt时刻的噪声
    else:
        predicted_noise = denoise_model(x_noisy, t, condition)
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    return loss


"""
  dataset && dataloader 
  Here we define a regular PyTorch Dataset. 
  The dataset simply consists of images from a real dataset, like Fashion-MNIST, CIFAR-10 or ImageNet, 
  scaled linearly to [−1, 1][−1,1].
  Each image is resized to the same size. 
   Interesting to note is that images are also randomly horizontally flipped. From the paper:
"""

# from datasets import load_dataset,load_from_disk

# load dataset from the hub
# dataset = load_from_disk('./data')
# image_size = 28
# channels = 1
# batch_size = 128
#
#
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose
#
# # define image transformations (e.g. using torchvision)
# transform = Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda t: (t * 2) - 1)
# ])
#
#
# # define function
# def transforms(examples):
#    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
#    del examples["image"]
#    return examples
#
#
# transformed_dataset = dataset.with_transform(transforms).remove_columns("label")
#
# # create dataloader
# dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
#
# batch = next(iter(dataloader))
# print(batch.keys())


"""
Generating new images from a diffusion model happens by reversing the diffusion process: 
we start from T, where we sample pure noise from a Gaussian distribution, 
and then use our neural network to gradually denoise it (using the conditional probability it has learned), 
until we end up at time step t = 0t=0. As shown above, we can derive a slighly less denoised image {x}_{t-1}
by plugging in the reparametrization of the mean, using our noise predictor.
Remember that the variance is known ahead of time.

Ideally, we end up with an image that looks like it came from the real data distribution.

"""


@torch.no_grad()
def p_sample(model, x, t, t_index, noise=None, condition=None):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    if condition is None:
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
    else:
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t, condition) / sqrt_one_minus_alphas_cumprod_t
        )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        if noise is None:
            noise = torch.randn_like(x)

        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


#
#     # Algorithm 2 (including returning all images)
#
#
@torch.no_grad()
def p_sample_loop(model, shape, img, condition=None):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    if img is None:
        img = torch.randn(shape, device=device)
    imgs = []

    # for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
    #     img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, condition=condition)
    #     imgs.append(img.cpu().numpy())
    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, condition=condition)
        imgs.append(img.cpu().numpy())
    return imgs


#
#
# @torch.no_grad()
# def sample(model, image_size, batch_size=16, channels=3):
#     return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

if __name__ == '__mian__':
    pass
