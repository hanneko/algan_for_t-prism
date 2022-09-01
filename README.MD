# ALGAN for T-PRISM
T-PRISMのためのALGAN実装

[T-PRISM：深層学習と論理プログラミングの融合](https://www.jstage.jst.go.jp/article/jjsai/36/4/36_437/_article/-char/ja/)\
小島 諒介, 佐藤 泰介


## How To Use This Code
You will need:

- [PyTorch](https://PyTorch.org/), version 1.8.1 later
- Recommendation: [NVIDIA Docker PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- The MNIST dataset

```sh
cd src
python main.py
```


後日、実行方法やコンテナでの起動方法、ディレクトリツリーなどを追記。

## 結果
MNISTの `0` を正常、 `1` を異常として学習。

![上段：正常潜在変数、下段：異常潜在変数](imgs/Gen_img_zdim100__epoch159_iter1440.png)

上段：正常潜在変数\
下段：異常潜在変数

![ヒストグラム](imgs/Test_histogram__iter_1440_epoch_159.png)

![AUROC](imgs/Test_ROC__iter_1440_epoch_159.png)


## Misc Notes
If you use this code, please cite
```text
@ARTICLE{murase_fukumizu_algan_2022,
author={Murase, Hironori and Fukumizu, Kenji},
journal={IEEE Access},
title={ALGAN: Anomaly Detection by Generating Pseudo Anomalous Data via Latent Variables},
year={2022},
volume={10},
number={},
pages={44259-44270},
doi={10.1109/ACCESS.2022.3169594}}
```

## Acknowledgments
This paper is based on results obtained from a project, JPNP16007, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).\
This work was supported in part by The Graduate University for Advanced Studies, SOKENDAI.