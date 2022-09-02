# -*- coding: utf-8 -*-
import torch


# 乱数のseed値
seed = 1234
TORCH_SEED = seed  # 1234
NP_SEED = seed  # 1234
RAND_SEED = seed  # 1234
SK_SEED = seed  # 1234

# GPUかCPUか # DEVICE = "cuda"  # "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ./src/ ディレクトリ内からプログラム実行を想定しているパス指定
OUTPUT_PATH = "../output/"
# ./src/ ディレクトリ内からプログラム実行を想定しているパス指定
DATASET_PATH = "../data"

NORMAL_CATEGOLY = 0  # 正常のカテゴリ（MNISTの数字）
ANOMALOUS_CATEGOLY = 1  # 異常のカテゴリ（MNISTの数字）
BATCH_SIZE_TRAIN = 512  # Trainingデータのミニバッチサイズ

# Validation, Testデータのミニバッチサイズ。
# 数字で性能は変化しない。大きいほど、Val, Testデータ全体を予測する時間が短くなる。
# GPUのメモリが少なくメモリ不足となる場合は小さな値を指定する。
BATCH_SIZE_VAL = 256
BATCH_SIZE_TEST = 256

# Trainingデータの Train-Val分割の比率。Training側の比率を指定する。
TRAIN_VAL_RATIO = 0.8

# MNISTデータセットの正規化定数
MEAN = [0.5]
STD = [0.5]

NUM_EPOCHS = 256  # 学習エポック数
VAL_EPOCH = 8  # VAL_EPOCHおきにValidationループへ入る

# 数字のエポック数の訓練はVALで高性能だったとしてもモデルをキープしない。
# Generatorの学習不十分にも関わらず、偶然にVALで良い値が仮に出てしまう場合は応急処置として0以外の値を指定する。
HP_PASS_EPOCH = 32

# Gen, Disの学習率
G_LR, D_LR = 0.0002, 0.0001
# Adamのベータ設定
BETA_1, BETA_2 = 0.0, 0.9  # 0.0, 0.9 or 0.5, 0.999 pytorch default: (0.9, 0.999)

####### ALGAN独自のハイパーパラメタ #######
Z_DIM = 100  # 潜在変数の次元
ANO_SIGMA = 4  # 異常潜在変数のシグマ
ALPHA = 0.75
XI = 0.75
N_Z = 2  # 2
N_DIS = 2  # 2

# データロード高速化の設定。ピュアなWindows環境で実行の場合は不具合が発生する。
# NUM_WORKERS = 4
# PIN_MEMORY = False
# DROP_LAST = True
# P_WORKER = True
CUDNN_BENCH = True
