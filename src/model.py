# -*- coding: utf-8 -*-

# import torch
import torch.nn as nn
# from torchvision import models  # transforms

# import config as c


class MLP_Generator(nn.Module):
    def __init__(self, z_dim=100, out_dim=784):
        super(MLP_Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(z_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            # nn.utils.spectral_norm(nn.Linear(256, 512)),  # 128, 256
            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024),  # 512
            nn.ReLU(inplace=True),
        )

        self.last = nn.Sequential(
            nn.Linear(1024, out_dim, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.last(out)

        return out


class MLP_Discriminator(nn.Module):
    def __init__(self, in_dim=784):
        super(MLP_Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(in_dim, 1024, bias=False)),  # in_dim, 256
            nn.LeakyReLU(0.2, inplace=True),
        )  # 0.1 → 0.2へ変更  #256->128

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(1024, 512, bias=False)),  # 256, 128
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )  # 0.1 → 0.2へ変更  #64

        self.last = nn.Sequential(
            # nn.utils.spectral_norm(nn.Linear(256, 1))
            nn.Linear(512, 1, bias=False)
        )  # 128, 1

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.last(out)

        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # nn.init.constant_(m.bias.data, 0)

    if classname.find("Linear") != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_uniform_(m.weight.data)
        # m.bias.data.fill_(0)
        # nn.init.constant_(m.bias.data, 0)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
