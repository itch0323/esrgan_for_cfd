import os
import os.path as osp
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from models import *
from datasets import *
class Opts():
    def __init__(self):
        self.n_epoch = 1
        self.residual_blocks = 23
        self.lr = 0.0002
        self.b1 = 0.9
        self.b2 = 0.999
        self.batch_size = 8
        self.n_cpu = 8
        self.warmup_batches = 5
        self.lambda_adv = 5e-3
        self.lambda_pixel = 1e-2
        self.pretrained = False
        self.dataset_name = 'cat'
        self.sample_interval = 100
        self.checkpoint_interval = 1000
        self.hr_height = 128
        self.hr_width = 128
        self.hr_shape = (128, 128)
        self.channels = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def to_dict(self):
        parameters = {
            'n_epoch': self.n_epoch,
            'hr_height': self.hr_height,
            'residual_blocks': self.residual_blocks,
            'lr': self.lr,
            'b1': self.b1,
            'b2': self.b2,
            'batch_size': self.batch_size,
            'n_cpu': self.n_cpu,
            'warmup_batches': self.warmup_batches,
            'lambda_adv': self.lambda_adv,
            'lambda_pixel': self.lambda_pixel,
            'pretrained': self.pretrained,
            'dataset_name': self.dataset_name,
            'sample_interval': self.sample_interval,
            'checkpoint_interval': self.checkpoint_interval,
            'hr_height': self.hr_height,
            'hr_width': self.hr_width,
            'hr_shape': self.hr_shape,
            'channels': self.channels,
            'device': str(self.device),
        }
        return parameters

def main():
    opt = Opts()
    print(opt.to_dict())
    hr_shape = (opt.hr_height, opt.hr_height)

    ROOT = 'esrgan/' #ルートディレクトリ content/esrgan/
    output_dir = osp.join(ROOT, 'output') # Colab内に保存 content/esrgan/output
    input_dir = osp.join(ROOT, 'input') # 入力となるデータセットを保存するディレクトリ content/esrgan/input
    image_dir = osp.join(input_dir, 'images') # 画像を保存するディレクトリ content/esrgan/input/images
    annotations_dir = osp.join(input_dir, 'annotations') # アノテーションデータを保存するディレクトリ content/esrgan/input/annotations
    list_path = osp.join(annotations_dir, 'list.txt') # アノテーションのリストファイルのパス content/esrgan/input/annotations/list.txt
    dataset_dir = osp.join(input_dir, 'cfd_sim') # データセットのディレクトリ content/esrgan/input/cat_face
    train_dir = osp.join(dataset_dir, 'train') # データセットの学習データを保存するディレクトリ content/esrgan/input/cat_face/train
    test_dir = osp.join(dataset_dir, 'test') # データセットのテストデータを保存するディレクトリ content/esrgan/input/cat_face/test
    demo_dir = osp.join(dataset_dir, 'demo') # デモ用のデータを保存するディレクトリ content/esrgan/input/cat_face/demo
    image_train_save_dir = osp.join(output_dir, 'image', 'train')
    image_test_save_dir = osp.join(output_dir, 'image', 'test')
    weight_save_dir = osp.join(output_dir, 'weight')
    param_save_path = osp.join(output_dir, 'param.json')
    log_dir = './logs'

    save_dirs = [input_dir, log_dir, image_train_save_dir, image_test_save_dir, weight_save_dir]
    for save_dir in save_dirs:
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    train_data_dir = osp.join(dataset_dir, 'train') #訓練データのディレクトリ content/esrgan/input/cat_face/train
    test_data_dir = osp.join(dataset_dir, 'test') #検証データのディレクトリ content/esrgan/input/cat_face/test
    demo_data_dir = osp.join(dataset_dir, 'demo') #テストデータのディレクトリ content/esrgan/input/cat_face/demo
    seed = 19930124

    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset_name = 'cat_face'

    train_dataloader = DataLoader(
        ImageDataset(train_data_dir, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    test_dataloader = DataLoader(
        TestImageDataset(test_data_dir),
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # ESRGANを呼び出す
    esrgan = ESRGAN(opt)

    for epoch in range(1, opt.n_epoch + 1):
        for batch_num, imgs in enumerate(train_dataloader):
            batches_done = (epoch - 1) * len(train_dataloader) + batch_num
            # 事前学習
            if batches_done <= opt.warmup_batches:
                esrgan.pre_train(imgs, batches_done, batch_num, epoch)
            # 本学習
            else:
                esrgan.train(imgs, batches_done, batch_num, epoch)

            # 学習した重みの保存
            if batches_done % opt.checkpoint_interval == 0:
                esrgan.save_weight(batches_done, weight_save_dir)
 
if __name__ == '__main__':
    main()