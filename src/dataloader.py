# -*- coding: utf-8 -*-

import os

import torch.utils.data as data
import torchvision
from torchvision import transforms

import config as c


def make_mnist_dataloader(train):
    nor_cat = c.NORMAL_CATEGOLY
    ano_cat = c.ANOMALOUS_CATEGOLY
    t_v_ratio = c.TRAIN_VAL_RATIO
    bs_train = c.BATCH_SIZE_TRAIN
    bs_val = c.BATCH_SIZE_VAL
    bs_test = c.BATCH_SIZE_TEST

    mean = c.MEAN
    std = c.STD

    path = c.DATASET_PATH  # "../data"
    is_dir = os.path.isdir(path)
    print("../data/ ディレクトリの存在 = {}".format(is_dir))

    # もし"/data"ディレクトリが無ければ作る
    # if not is_dir:
    if is_dir:
        pass
    else:
        os.makedirs(path)

    dataset_nor = torchvision.datasets.MNIST(
        root=path,
        train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        download=True
    )

    dataset_ano = torchvision.datasets.MNIST(
        root=path,
        train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        download=True
    )

    # NormalとAnomalousのクラス指定
    mask_nor = dataset_nor.targets == nor_cat
    mask_ano = dataset_ano.targets == ano_cat
    dataset_nor.data = dataset_nor.data[mask_nor]
    dataset_nor.targets = dataset_nor.targets[mask_nor]
    dataset_ano.data = dataset_ano.data[mask_ano]
    dataset_ano.targets = dataset_ano.targets[mask_ano]

    # dataset sprit, train -> train-val
    if train:  # Train-Val
        n_samples_nor = len(dataset_nor)
        n_samples_ano = len(dataset_ano)
        train_size_nor = int(len(dataset_nor) * t_v_ratio)
        train_size_ano = int(len(dataset_ano) * t_v_ratio)
        val_size_nor = n_samples_nor - train_size_nor
        val_size_ano = n_samples_ano - train_size_ano

        # train-valへのデータセット分割
        dataset_nor_train, dataset_nor_val = data.random_split(dataset_nor, [train_size_nor, val_size_nor])
        _, dataset_ano_val = data.random_split(dataset_ano, [train_size_ano, val_size_ano])

        # data loaderの作成
        dataloader_train = data.DataLoader(dataset=dataset_nor_train, batch_size=bs_train, shuffle=True, drop_last=True)
        dataloader_val_nor = data.DataLoader(dataset=dataset_nor_val, batch_size=bs_val, shuffle=False, drop_last=False)
        dataloader_val_ano = data.DataLoader(dataset=dataset_ano_val, batch_size=bs_val, shuffle=False, drop_last=False)

        dataloaders = [dataloader_train, dataloader_val_nor, dataloader_val_ano]

    else:  # Test
        dataloader_test_nor = data.DataLoader(dataset=dataset_nor, batch_size=bs_test, shuffle=False, drop_last=False)
        dataloader_test_ano = data.DataLoader(dataset=dataset_ano, batch_size=bs_test, shuffle=False, drop_last=False)

        dataloaders = [dataloader_test_nor, dataloader_test_ano]

    return dataloaders


class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)
