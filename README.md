# super resolution
pytorchを使用して2次元キャビティ流れの数値計算から得たデータの超解像を実行できます。

このリポジトリではESRGAN<sup>1)</sup>をサポートしています。

実行環境と使用方法は下記の通りです。

<br>

## 実行環境(env)
- Python 3.7.9
- Pytorch 1.3.1
- Numpy 1.19.2
- Opencv 3.4.2
- Matplotlib 3.3.2
- Pillow 8.0.1

<br>

## 使用方法(usage)
1. リポジトリをクローンします。
```
git clone https://github.com/itch0323/esrgan_for_cfd.git
cd super_resolution
```

<br>

2. ディレクトリを移動し、anacondaで作成した環境を読み込んで切り替えます。
```
conda env create sr -f=sr.yml
conda activate pytorch
```

<br>

3. データセットを作成します。作成が終わるまで2分程度の時間を要します。
```
sh setup.sh
```

4. Jupyter Notebookにて超解像を実行します。
- ESRGAN
```
jupyter-notebook
```
<br>


## 論文
<p>1) Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao, Chen Change Loy.<a href="https://arxiv.org/pdf/1809.00219.pdf">　Enhanced Super-Resolution Generative Adversarial Networks</a>