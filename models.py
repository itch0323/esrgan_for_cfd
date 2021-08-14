import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg19
import sys
import json

from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

def mse(y, t):
    return 0.5 * np.sum((y - t) **2)

class DenseResidualBlock(nn.Module):
    """
    GenearatorのDenseResidualBlockのクラス
    """
    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale
        
        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)
    
        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
    
    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x

class ResidualInResidualDenseBlock(nn.Module):
    """
    GenearatorのResidualInResidualDenseBlockのクラス
    """
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), 
            DenseResidualBlock(filters), 
            DenseResidualBlock(filters)
        )
    
    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class GeneratorRRDB(nn.Module):
    """
    Generatorのクラス
    """
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) 
                                        for _ in range(num_res_blocks)])
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        
        upsample_layers = []
        
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class FeatureExtractor(nn.Module):
    """
    Perceputual lossを計算するために特徴量を抽出するためのクラス
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(
            vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)

class Discriminator(nn.Module):
    """
    Discriminatorのクラス
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
                
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)
    
        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, 
                                    out_filters, 
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, 
                                    out_filters, 
                                    kernel_size=3, 
                                    stride=2, 
                                    padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            print(discriminator_block(in_filters, 
                                      out_filters, 
                                      first_block=(i == 0)))
            layers.extend(discriminator_block(in_filters, 
                                              out_filters, 
                                              first_block=(i == 0)))
            in_filters = out_filters
        
        layers.append(nn.Conv2d(out_filters, 
                                1, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, img):
        return self.model(img)

class ESRGAN():
    """
    ESRGANの処理を実装するクラス
    """
    def __init__(self, opt):
        self.generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(opt.device)
        self.discriminator = Discriminator(input_shape=(opt.channels, *opt.hr_shape)).to(opt.device)

        self.feature_extractor = FeatureExtractor().to(opt.device)
        self.feature_extractor.eval()

        self.criterion_GAN = nn.BCEWithLogitsLoss().to(opt.device)
        self.criterion_content = nn.L1Loss().to(opt.device)
        self.criterion_pixel = nn.L1Loss().to(opt.device)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.writer = SummaryWriter(log_dir='./logs')
        self.lambda_adv = opt.lambda_adv
        self.lambda_pixel = opt.lambda_pixel
    
    def pre_train(self, imgs, batches_done, batch_num, epoch):
        """
        loss pixelのみで事前学習を行う
        """
        with torch.no_grad():
            # preprocess
            imgs_lr = Variable(imgs['lr'].type(self.Tensor))
            imgs_hr = Variable(imgs['hr'].type(self.Tensor))

            # ground truth
            valid = Variable(self.Tensor(np.ones((imgs_lr.size(0), *self.discriminator.output_shape))), 
                              requires_grad=False)
            fake = Variable(self.Tensor(np.zeros((imgs_lr.size(0), *self.discriminator.output_shape))), 
                            requires_grad=False)

        # バックプロパゲーションの前に勾配を0にする
        self.optimizer_G.zero_grad()


        # 低解像度の画像から高解像度の画像を生成
        gen_hr = self.generator(imgs_lr)

        loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)

        # 画素単位の損失であるloss_pixelで事前学習を行う
        loss_pixel.backward()
        self.optimizer_G.step()
        train_info = {'epoch': epoch, 'batch_num': batch_num, 'loss_pixel': loss_pixel.item()}
        if batch_num == 1:
            sys.stdout.write('\n{}'.format(train_info))
        else:
            sys.stdout.write('\r{}'.format('\t'*20))
            sys.stdout.write('\r{}'.format(train_info))
        sys.stdout.flush()

        self.save_loss(train_info, batches_done)

    def train(self, imgs, batches_done, batch_num, epoch):
        """
        pixel loss以外の損失も含めて本学習を行う
        """
        with torch.no_grad():
            # 前処理
            imgs_lr = Variable(imgs['lr'].type(self.Tensor))
            imgs_hr = Variable(imgs['hr'].type(self.Tensor))

            # 正解ラベル
            valid = Variable(self.Tensor(np.ones((imgs_lr.size(0), *self.discriminator.output_shape))), 
                              requires_grad=False)
            fake = Variable(self.Tensor(np.zeros((imgs_lr.size(0), *self.discriminator.output_shape))), 
                            requires_grad=False)

        # 低解像度の画像から高解像度の画像を生成
        self.optimizer_G.zero_grad()
        gen_hr = self.generator(imgs_lr)

        # Pixel loss
        loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)

        # 推論
        pred_real = self.discriminator(imgs_hr).detach()
        pred_fake = self.discriminator(gen_hr)

        # Adversarial loss
        loss_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Perceptual loss
        gen_feature = self.feature_extractor(gen_hr)
        real_feature = self.feature_extractor(imgs_hr).detach()
        loss_content = self.criterion_content(gen_feature, real_feature)

        # 生成器のloss
        loss_G = loss_content + self.lambda_adv * loss_GAN + self.lambda_pixel * loss_pixel
        loss_G.backward()
        self.optimizer_G.step()

        # 識別機のLoss
        self.optimizer_D.zero_grad()
        pred_real = self.discriminator(imgs_hr)
        pred_fake = self.discriminator(gen_hr.detach())

        loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)            
        loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)    
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        self.optimizer_D.step()

        train_info = {'epoch': epoch, 'batch_num': batch_num,  'loss_D': loss_D.item(), 'loss_G': loss_G.item(),
                      'loss_content': loss_content.item(), 'loss_GAN': loss_GAN.item(), 'loss_pixel': loss_pixel.item(),}
        if batch_num == 1:
            sys.stdout.write('\n{}'.format(train_info))
        else:
            sys.stdout.write('\r{}'.format('\t'*20))
            sys.stdout.write('\r{}'.format(train_info))
        sys.stdout.flush()

        self.save_loss(train_info, batches_done)

    def save_loss(self, train_info, batches_done):
        """
        lossの保存
        """
        for k, v in train_info.items():
            self.writer.add_scalar(k, v, batches_done)

    def save_weight(self, batches_done, weight_save_dir):
        """
        重みの保存
        """
        # Save model checkpoints
        generator_weight_path = osp.join(weight_save_dir, "generator_{:08}.pth".format(batches_done))
        discriminator_weight_path = osp.join(weight_save_dir, "discriminator_{:08}.pth".format(batches_done))

        torch.save(self.generator.state_dict(), generator_weight_path)
        torch.save(self.discriminator.state_dict(), discriminator_weight_path)

def save_json(file, save_path, mode):
    """Jsonファイルを保存
    """
    with open(save_path, mode) as outfile:
        json.dump(file, outfile, indent=4)

