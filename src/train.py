# -*- coding: utf-8 -*-

import random
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

import config as c
from utils import from_buffer, latent_variable_generator, save_fake_images, to_buffer

# Setup seeds
torch.manual_seed(c.TORCH_SEED)
np.random.seed(c.NP_SEED)
random.seed(c.RAND_SEED)


def train_full_gan(D, G, dataloaders_train_val, dataloaders_test):
    time_train_start = time.time()

    dataloader_train = dataloaders_train_val[0]
    dataloaders_val = [dataloaders_train_val[1], dataloaders_train_val[2]]
    dataloaders_test = dataloaders_test

    bs_NOR = c.BATCH_SIZE_TRAIN
    bs_ANO = c.BATCH_SIZE_TRAIN

    D_optimizer = torch.optim.Adam(D.parameters(), c.D_LR, (c.BETA_1, c.BETA_2))
    G_optimizer = torch.optim.Adam(G.parameters(), c.G_LR, (c.BETA_1, c.BETA_2))

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # 変数初期化
    val_auc_max = 0.0
    fake_buf_NOR, fake_buf_ANO = None, None
    D_real_loss = torch.tensor(0.0)
    D_fake_NOR_loss, D_fake_NOR_loss = torch.tensor(0.0), torch.tensor(0.0)
    D_buf_NOR_loss, D_buf_ANO_loss = torch.tensor(0.0), torch.tensor(0.0)
    G_loss = torch.tensor(0.0)
    real_acc, fake_NOR_acc, fake_ANO_acc, fake_buf_NOR_acc, fake_buf_ANO_acc = 0.0, 0.0, 0.0, 0.0, 0.0
    loss_log, acc_log = np.empty(0), np.empty(0)

    time_epoch_list = list()

    # 画像で可視化するための潜在変数（固定）
    fixed_z_NOR, fixed_z_ANO = latent_variable_generator(8, 8)  # 正常8枚、異常8枚
    fixed_z_NOR, fixed_z_ANO = fixed_z_NOR.to(c.DEVICE), fixed_z_ANO.to(c.DEVICE)

    print("######## Start Training ########")
    iteration = 0

    for epoch in range(c.NUM_EPOCHS):
        time_epoch_start = time.time()
        D_loss_epoch, D_loss_buf_epoch, G_loss_epoch = 0.0, 0.0, 0.0
        print("Epoch {}/{}".format(epoch, c.NUM_EPOCHS))

        for imgs_NOR, _label in dataloader_train:
            # 潜在変数のサンプル(c.N_Z毎にサンプル)
            if epoch % c.N_Z == 0:
                z_NOR, z_ANO = latent_variable_generator(bs_NOR, bs_ANO)
                z_NOR_gen, _ = latent_variable_generator(c.BATCH_SIZE_TRAIN, bs_ANO)
                z_NOR, z_ANO, z_NOR_gen = z_NOR.to(c.DEVICE), z_ANO.to(c.DEVICE), z_NOR_gen.to(c.DEVICE)

            # ミニバッチがデータセットの端数のときは学習を飛ばす
            if imgs_NOR.size()[0] != c.BATCH_SIZE_TRAIN:
                continue
            
            # ミニバッチをGPUへ
            imgs_NOR = imgs_NOR.view(-1, 28*28).to(c.DEVICE)
            # ----------------------
            # Discriminatorの訓練
            # ----------------------
            for _ndis_count in range(c.N_DIS):
                D_optimizer.zero_grad()
                G_optimizer.zero_grad()

                # Real-Normal の識別 -> 1 Real
                D_real = D(imgs_NOR.detach())

                if epoch == 0:  # 1エポック目はバッファが無い
                    # Fake_NOR の識別 -> 0 Fake
                    fake_new_NOR = G(z_NOR)
                    D_fake_NOR = D(fake_new_NOR.detach())

                    # Fake_ANO の生成と識別 -> 0 Fake
                    G.eval()  # Gを.eval()にしないとBatchNormのパラメタが壊れる
                    fake_new_ANO = G(z_ANO)
                    fake_new_ANO = fake_new_ANO.detach()
                    G.train()  # Gを.train()に戻す
                    D_fake_ANO = D(fake_new_ANO)

                    # ロスの計算
                    D_real_loss = criterion(D_real.view(-1), torch.full((D_real.size(0),), 1.0).to(c.DEVICE))  # real = 1.0
                    D_fake_NOR_loss = criterion(D_fake_NOR.view(-1), torch.full((D_fake_NOR.size(0),), 0.0).to(c.DEVICE))  # fake 0.0
                    D_fake_ANO_loss = criterion(D_fake_ANO.view(-1), torch.full((D_fake_ANO.size(0),), 0.0).to(c.DEVICE))  # fake 0.0

                    # ロスの合算
                    D_loss = D_real_loss + c.ALPHA*D_fake_NOR_loss + (1.0-c.ALPHA)*D_fake_ANO_loss
                    D_loss.backward()
                    D_optimizer.step()

                else:  # 2エポック目以降はバッファがある
                    # Fake_NOR の識別 -> 0 Fake
                    fake_new_NOR = G(z_NOR)
                    D_fake_NOR = D(fake_new_NOR.detach())

                    # Fake_ANO の生成と識別 -> 0 Fake
                    G.eval()  # Gを.eval()にしないとBatchNormのパラメタが壊れる
                    fake_new_ANO = G(z_ANO)
                    fake_new_ANO = fake_new_ANO.detach()
                    G.train()  # Gを.train()に戻す
                    D_fake_ANO = D(fake_new_ANO)

                    # バッファをサンプル
                    fake_buf_NOR_pred = from_buffer(fake_buf_NOR, num_of_imgs=bs_NOR)
                    fake_buf_ANO_pred = from_buffer(fake_buf_ANO, num_of_imgs=bs_ANO)

                    # Buf_NOR と Buf_ANO の識別 -> 0 Fake
                    D_buf_NOR = D(fake_buf_NOR_pred)
                    D_buf_ANO = D(fake_buf_ANO_pred)

                    # ロスの計算
                    D_real_loss = criterion(D_real.view(-1), torch.full((D_real.size(0),), 1.0).to(c.DEVICE))  # real = 1.0
                    D_fake_NOR_loss = criterion(D_fake_NOR.view(-1), torch.full((D_fake_NOR.size(0),), 0.0).to(c.DEVICE))  # fake 0.0
                    D_fake_ANO_loss = criterion(D_fake_ANO.view(-1), torch.full((D_fake_ANO.size(0),), 0.0).to(c.DEVICE))  # fake 0.0
                    D_buf_NOR_loss = criterion(D_buf_NOR.view(-1), torch.full((D_buf_NOR.size(0),), 0.0).to(c.DEVICE))  # fake = 0.0
                    D_buf_ANO_loss = criterion(D_buf_ANO.view(-1), torch.full((D_buf_ANO.size(0),), 0.0).to(c.DEVICE))  # fake = 0.0

                    # ロスの合算
                    D_loss_buf = D_real_loss + c.ALPHA*(c.XI*D_fake_NOR_loss + (1.0-c.XI)*D_buf_NOR_loss) + (1.0-c.ALPHA)*(c.XI*D_fake_ANO_loss + (1.0-c.XI)*D_buf_ANO_loss)

                    D_loss_buf.backward()
                    D_optimizer.step()

            # ----------------------
            # Generatorの訓練
            # ----------------------
            D_optimizer.zero_grad()
            G_optimizer.zero_grad()

            # Fake-NOR の識別 -> 1
            D_fake_out = D(G(z_NOR_gen))

            # ロスの計算
            G_loss = criterion(D_fake_out.view(-1), torch.full((D_fake_out.size(0),), 1.0).to(c.DEVICE))

            G_loss.backward()
            G_optimizer.step()

            D_optimizer.zero_grad()
            G_optimizer.zero_grad()

            # ----------------------
            # Accuracyの計算
            # ----------------------
            if epoch >= 1:
                fake_buf_NOR_acc = len(D_buf_NOR.view(-1)[D_buf_NOR.view(-1) < 0.0]) / len(D_buf_NOR.view(-1))
                fake_buf_ANO_acc = len(D_buf_ANO.view(-1)[D_buf_ANO.view(-1) < 0.0]) / len(D_buf_ANO.view(-1))

            real_acc = len(D_real.view(-1)[D_real.view(-1) >= 0.0]) / len(D_real.view(-1))
            fake_NOR_acc = len(D_fake_NOR.view(-1)[D_fake_NOR.view(-1) < 0.0]) / len(D_fake_NOR.view(-1))
            fake_ANO_acc = len(D_fake_ANO.view(-1)[D_fake_ANO.view(-1) < 0.0]) / len(D_fake_ANO.view(-1))

            print(
                "iter {} |epoch {} |R_Acc:{:.2f} | F_NOR_Acc:{:.2f} | F_ANO_Acc:{:.2f} | F_Bf_NOR_Acc:{:.2f} | F_Bf_ANO_Acc:{:.2f}".format(
                    iteration, epoch, real_acc, fake_NOR_acc, fake_ANO_acc, fake_buf_NOR_acc, fake_buf_ANO_acc)
            )

            # ----------------------
            # バッファの保存
            # ----------------------
            fake_buf_NOR = to_buffer(fake_buf_NOR, fake_new_NOR, buf_per_iter=bs_NOR)
            fake_buf_ANO = to_buffer(fake_buf_ANO, fake_new_ANO, buf_per_iter=bs_ANO)

            # ----------------------
            # ログの記録
            # ----------------------
            G_loss_epoch += G_loss.item()
            D_loss_epoch += D_loss.item()
            if epoch >= 1:
                D_loss_buf_epoch += D_loss_buf.item()

            # ロスの保存
            if iteration == 0:
                loss_log = np.array(
                    [
                        iteration,
                        epoch,
                        D_real_loss.cpu().detach().numpy(),
                        D_fake_NOR_loss.cpu().detach().numpy(),
                        D_fake_ANO_loss.cpu().detach().numpy(),
                        D_buf_NOR_loss.cpu().detach().numpy(),
                        D_buf_ANO_loss.cpu().detach().numpy(),
                        G_loss.cpu().detach().numpy(),
                    ]
                ).astype(np.float32)
            else:
                iter_loss = np.array(
                    [
                        iteration,
                        epoch,
                        D_real_loss.cpu().detach().numpy(),
                        D_fake_NOR_loss.cpu().detach().numpy(),
                        D_fake_ANO_loss.cpu().detach().numpy(),
                        D_buf_NOR_loss.cpu().detach().numpy(),
                        D_buf_ANO_loss.cpu().detach().numpy(),
                        G_loss.cpu().detach().numpy(),
                    ]
                ).astype(np.float32)
                loss_log = np.vstack((loss_log, iter_loss))

            # Accuracyの保存
            if iteration == 0:
                acc_log = np.array(
                    [
                        iteration,
                        epoch,
                        real_acc,
                        fake_NOR_acc,
                        fake_ANO_acc,
                        fake_buf_NOR_acc,
                        fake_buf_ANO_acc,
                    ]
                ).astype(np.float32)
            else:
                iter_acc = np.array(
                    [
                        iteration,
                        epoch,
                        real_acc,
                        fake_NOR_acc,
                        fake_ANO_acc,
                        fake_buf_NOR_acc,
                        fake_buf_ANO_acc,
                    ]
                ).astype(np.float32)
                acc_log = np.vstack((acc_log, iter_acc))

            iteration += 1

        # 学習時間の計算
        time_epoch_finish = time.time()
        epoch_finish = time_epoch_finish - time_epoch_start
        print("Epoch calc time:  {:.4f} sec.".format(epoch_finish))
        time_epoch_list.append(epoch_finish)

        # ----------------------
        # 正常画像と異常画像の保存
        # ----------------------
        if (epoch + 1) % c.VAL_EPOCH == 0:
            G.eval()
            with torch.no_grad():
                fixed_fake_NOR = G(fixed_z_NOR)
                fixed_fake_ANO = G(fixed_z_ANO)
            G.train()
            save_fake_images(fixed_fake_NOR.detach(), fixed_fake_ANO.detach(), iteration, epoch)

        # VAL_EPOCH毎にValidationする
        if (epoch + 1) % c.VAL_EPOCH == 0:
            G.eval()
            D.eval()
            val_auc = predict_full_gan(D, dataloaders_val, iteration, epoch, val=True)  # Validation

            # ValidationのAUCが最大のとき、モデルをキープする
            if val_auc >= val_auc_max and (epoch + 1) >= c.HP_PASS_EPOCH:
                val_auc_max = val_auc
                D_high_perf = deepcopy(D)
                G_high_perf = deepcopy(G)
                high_perf_epoch, high_perf_iteration = epoch, iteration
            G.train()
            D.train()

        # エポック毎のロスとAccuracy
        print(
            "epoch {} |D_loss:{:.4f} |D_loss_buf:{:.4f} | G_loss:{:.4f}".format(
                epoch, D_loss_epoch, D_loss_buf_epoch, G_loss_epoch
            )
        )

    # Test dataの評価と最も性能の高かったモデルの保存。
    G.eval()
    D.eval()
    D_high_perf.eval()
    G_high_perf.eval()

    _ = predict_full_gan(D_high_perf, dataloaders_test, high_perf_iteration, high_perf_epoch, val=False)

    # モデルの保存
    D_name = c.OUTPUT_PATH + "Dis_last_epoch_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".pth"
    G_name = c.OUTPUT_PATH + "Gen_last_epoch_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".pth"
    D_high_perf_name = c.OUTPUT_PATH + "Dis_highest_performance_" + "_iter_" + str(high_perf_iteration) + "_epoch_" + str(high_perf_epoch) + ".pth"
    G_high_perf_name = c.OUTPUT_PATH + "Gen_highest_performance_" + "_iter_" + str(high_perf_iteration) + "_epoch_" + str(high_perf_epoch) + ".pth"
    torch.save(D.module.state_dict(), D_name)
    torch.save(G.module.state_dict(), G_name)
    torch.save(D_high_perf.module.state_dict(), D_high_perf_name)
    torch.save(G_high_perf.module.state_dict(), G_high_perf_name)
    print("saved models at last epoch")

    # loss logの保存
    loss_log_file_name = c.OUTPUT_PATH + "loss_log_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".csv"
    np.savetxt(loss_log_file_name, loss_log, delimiter=",")

    # acc logの保存
    acc_log_file_name = c.OUTPUT_PATH + "acc_log_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".csv"
    np.savetxt(acc_log_file_name, acc_log, delimiter=",")

    # bufferの削除
    fake_buf_NOR = None
    fake_buf_ANO = None
    loss_log = None
    acc_log = None

    time_train_finish = time.time()
    print("total calc. time: {:.4f} sec.".format(time_train_finish - time_train_start))

    epoch_file_name = c.OUTPUT_PATH + "Epoch_time_" + ".txt"
    time_file = open(epoch_file_name, "w")
    for time_row in time_epoch_list:
        time_file.write(str(time_row) + "\n")
    time_file.close()


def predict_full_gan(D_trained, data_loader_list, iteration, epoch, val=True):
    """スコアを計算しラベルと比較する。ヒストグラムとROC曲線を描画しAUCを計算する。"""
    # 変数初期化
    score, score_normal, score_anomalous = np.empty(0), np.empty(0), np.empty(0)
    gt_label_1, gt_label_0 = np.empty(0), np.empty(0)

    # ラベルのディショナリ
    label_list = ["good", "anomaly"]

    time_pred_start = time.time()
    for i, label in enumerate(label_list):  # 正常と異常のラベルでループする
        data_loader = data_loader_list[i]
        print("category is {}".format(label))
        
        for imgs, _label in data_loader:
            print("Number of images = {}".format(imgs.size()))
            with torch.no_grad():
                score_single = D_trained(imgs.view(-1, 28*28).to(c.DEVICE))
            score = np.append(score, -1 * score_single.cpu().detach().numpy())
            # スコアのセーブ。テストのラベルを作る。
            # 2クラスのラベルを返す。１のラベルは異常、０のラベルは正常。
            if label == "good":
                score_normal = np.append(score_normal, score)
                label_0 = np.zeros(imgs.size()[0], dtype="int8")
                gt_label_0 = np.append(gt_label_0, label_0)

            else:  # label == "anomalous"
                score_anomalous = np.append(score_anomalous, score)
                label_1 = np.ones(imgs.size()[0], dtype="int8")
                gt_label_1 = np.append(gt_label_1, label_1)

            score = np.empty(0)

    time_pred_end = time.time()
    pred_time = time_pred_end - time_pred_start
    print("total pred time: {}".format(pred_time))
    pred_time_list = [pred_time]

    # 保存
    pred_file_name = c.OUTPUT_PATH +  "pred_time_" + "_epoch" + str(epoch) + ".txt"
    time_file = open(pred_file_name, "w")
    for time_row in pred_time_list:
        time_file.write(str(time_row) + "\n")
    time_file.close()

    if val:
        score_file_name = c.OUTPUT_PATH + "Val_score_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".txt"
        gt_label_file_name = c.OUTPUT_PATH + "Val_gt_label_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".txt"
        fig_hist_name = c.OUTPUT_PATH + "Val_histogram_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".png"
        fig_roc_name = c.OUTPUT_PATH + "Val_ROC_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".png"
        auc_file_name = c.OUTPUT_PATH + "Val_AUC_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".txt"
    else:  # test
        score_file_name = c.OUTPUT_PATH + "Test_score_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".txt"
        gt_label_file_name = c.OUTPUT_PATH + "Test_gt_label_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".txt"
        fig_hist_name = c.OUTPUT_PATH + "Test_histogram_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".png"
        fig_roc_name = c.OUTPUT_PATH + "Test_ROC_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".png"
        auc_file_name = c.OUTPUT_PATH + "Test_AUC_" + "_iter_" + str(iteration) + "_epoch_" + str(epoch) + ".txt"

    # スコアのlistを作成
    score_anomalous_normal = np.concatenate((score_anomalous, score_normal), axis=0)
    # スコアのlistを保存
    score_file = open(score_file_name, "w")
    for score_row in score_anomalous_normal:
        score_file.write(str(score_row) + "\n")
    score_file.close()

    # GTラベルのlistを作成。１のラベルが異常、０のラベルが正常。
    label_anomalous_normal = np.concatenate((gt_label_1, gt_label_0), axis=0)
    # ラベルのlistを保存
    gt_label_file = open(gt_label_file_name, "w")
    for gt_label_row in label_anomalous_normal:
        gt_label_file.write(str(gt_label_row) + "\n")
    gt_label_file.close()

    # min, maxの軸
    axis_min = min(score_anomalous_normal)
    axis_max = max(score_anomalous_normal)
    # ヒストグラム
    figure_histogram = plt.figure()
    plt.hist(
        score_anomalous,
        bins=20,
        alpha=0.5,
        histtype="stepfilled",
        range=(axis_min, axis_max),
        label="anomalous",
        color=["#ff7f0e"]
    )
    plt.hist(
        score_normal,
        bins=20,
        alpha=0.5,
        histtype="stepfilled",
        range=(axis_min, axis_max),
        label="normal",
        color=["#1f77b4"]
    )
    plt.legend(loc="upper left")
    # save
    figure_histogram.savefig(fig_hist_name)
    plt.close(figure_histogram)

    # ROC
    fpr, tpr, thresholds = roc_curve(label_anomalous_normal, score_anomalous_normal)
    figure_roc = plt.figure()
    plt.plot(fpr, tpr, marker="o")
    plt.xlabel("FPR: False positive rate")
    plt.ylabel("TPR: True positive rate")
    plt.grid()
    # save
    figure_roc.savefig(fig_roc_name)
    plt.close(figure_roc)

    # AUC
    auc = roc_auc_score(label_anomalous_normal, score_anomalous_normal)
    # save
    auc_file = open(auc_file_name, "w")
    auc_file.write(str(auc))
    auc_file.close()

    return auc
