# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import config as c
from dataloader import make_mnist_dataloader
from model import MLP_Discriminator, MLP_Generator, weights_init
from train import train_full_gan

if __name__ == "__main__":

    z_dim = c.Z_DIM

    G = MLP_Generator(z_dim=z_dim, out_dim=784)  # out_dim=512
    D = MLP_Discriminator(in_dim=784)  # in_dim=512

    G.apply(weights_init).to(c.DEVICE).train()
    D.apply(weights_init).to(c.DEVICE).train()

    G = nn.DataParallel(G, device_ids=[0])
    D = nn.DataParallel(D, device_ids=[0])

    print("Gen. on => {}".format(next(G.parameters()).is_cuda))
    print("Dis. on => {}".format(next(D.parameters()).is_cuda))

    if c.DEVICE == "cuda":
        torch.backends.cudnn.benchmark = c.CUDNN_BENCH

    # create Data Loader

    dataloaders_train_val = make_mnist_dataloader(train=True)
    dataloaders_test = make_mnist_dataloader(train=False)

    # start training
    train_full_gan(D, G, dataloaders_train_val, dataloaders_test)

    print("finish training!!")

