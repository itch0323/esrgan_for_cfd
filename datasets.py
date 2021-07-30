import glob
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, dataset_dir, hr_shape):
        dataset_lrdir =osp.join(dataset_dir, "bin_32")
        dataset_hrdir =osp.join(dataset_dir, "bin_128")

        self.lr_files = sorted(glob.glob(osp.join(dataset_lrdir, "*")))
        self.hr_files = sorted(glob.glob(osp.join(dataset_hrdir, "*")))

    def __getitem__(self, index):
        img_lr = torch.from_numpy((np.load(self.lr_files[index % len(self.lr_files)])).astype(np.float32)).clone()
        img_hr = torch.from_numpy((np.load(self.hr_files[index % len(self.hr_files)])).astype(np.float32)).clone()

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.lr_files)
