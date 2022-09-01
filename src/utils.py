# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchvision

import config as c


def latent_variable_generator(bs_nor, bs_ano):

    nor_mu = 0
    nor_sigma = 1
    ano_mu = 0
    ano_sigma = c.ANO_SIGMA
    z_dim = c.Z_DIM

    # 正常 潜在変数（正規分布）
    z_nor = np.random.normal(nor_mu, nor_sigma, size=(bs_nor, z_dim)).astype(np.float32)
    z_nor = torch.from_numpy(z_nor)

    # 異常 潜在変数（正規分布）
    z_ano = np.random.normal(ano_mu, ano_sigma, size=(bs_ano, z_dim)).astype(np.float32)
    z_ano = torch.from_numpy(z_ano)

    return z_nor, z_ano


def to_buffer(buffer, new_fake, buf_per_iter):
    perm = torch.randperm(new_fake.size()[0])
    idx = perm[:buf_per_iter]
    new_fake = new_fake.detach()
    new_buffer = new_fake[idx]

    if buffer is None:
        print("no buffer")
        buffer = new_buffer
    else:
        buffer = torch.cat((buffer, new_buffer))
        # new
        perm = torch.randperm(buffer.size()[0])
        idx = perm[: buf_per_iter * 2]
        buffer = buffer.detach()
        buffer = buffer[idx]

    return buffer


def from_buffer(buffer, num_of_imgs):
    perm = torch.randperm(buffer.size()[0])
    idx = perm[:num_of_imgs]
    buffered_fake_images = buffer[idx]
    return buffered_fake_images


def save_fake_images(fixed_fake_NOR, fixed_fake_ANO, iteration, epoch):
    """fakeの画像をtorch_tensorから直接PNGに保存する"""
    gen_img_file_name = (c.OUTPUT_PATH + "Gen_img_" + "zdim" + str(c.Z_DIM) + "_" + "_epoch" + str(epoch) + "_iter" + str(iteration) + ".png")
    print("fixed_fake_NOR size = {}".format(fixed_fake_NOR.size()))
    visualize_fake_images_NOR = fixed_fake_NOR.detach()
    print("visualize_fake_images_NOR size = {}".format(visualize_fake_images_NOR.size()))
    visualize_fake_images_ANO = fixed_fake_ANO.detach()
    visualize_fake_images = torch.cat((visualize_fake_images_NOR, visualize_fake_images_ANO))
    print("visualize_fake_images size = {}".format(visualize_fake_images.size()))
    visualize_fake_images = visualize_fake_images.view(-1, 1, 28, 28)
    print("visualize_fake_images size = {}".format(visualize_fake_images.size()))
    torchvision.utils.save_image(
        visualize_fake_images,
        gen_img_file_name,
        nrow=8,
        padding=4,
        normalize=True,
        value_range=(-1.0, 1.0),
        pad_value=255,
    )
