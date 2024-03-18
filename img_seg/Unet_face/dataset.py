import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import color
from torchvision import transforms

# train_img_path = 'C:/Users/jun/Downloads/dataset/training/'
# train_img_num = os.listdir(train_img_path)
#
#
# train_img_fp = os.path.join(train_img_path, train_img_num[0])
#
# print(train_img_fp)
# print(np.array(Image.open(train_img_fp)).shape)
#
# a = np.array(Image.open(train_img_fp))
# print(a.ndim)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.train_img_path = data_dir
        self.transform = transform
        self.train_img_num = os.listdir(self.train_img_path)

    def __len__(self):
        return len(self.train_img_num)

    def __getitem__(self, index):
        train_img_fp = os.path.join(self.train_img_path, self.train_img_num[index])
        ## 내일 해당 부분부터 다시 짜야 함.
        ## AttributeError: 'Dataset' object has no attribute 'train_matte_path'. Did you mean: 'train_img_path'?
        # 이와 같은 에러가 발생했으므로 해결을 해야함.

        if 'matte' in train_img_fp:

        else:
        train_matte_fp = os.path.join(self.train_matte_path, self.train_matte_num[index])
        train_img, train_matte = self.train_img_dataset(train_img_fp, train_matte_fp)

        train_img = train_img / 255.0
        train_matte = train_matte / 255.0
        train_img = train_img.astype(np.float32)
        train_matte = train_matte.astype(np.float32)

        if train_img.ndim == 2:
            train_img = train_img[..., np.newaxis]
        if train_matte.ndim == 2:
            train_matte = train_matte[..., np.newaxis]

        data = {"train":train_img, "target":train_matte}

        if self.transform:
            data = self.transfrom(data)

        return data

    def train_img_dataset(self, train_img_path, train_matte_path):
        train_images = []
        train_mattes = []

        train_images.append(np.array(Image.open(train_img_path)))
        train_mattes.append(np.array(Image.open(train_matte_path)))

        return train_images, train_mattes


class ToTensor(object):
    def __call__(self, data):
        train, target = data['train'], data['target']

        train = train.transpose((2, 0, 1)).astype(np.float32)
        target = target.transpose((2, 0, 1)).astype(np.float32)

        data = {'train':torch.from_numpy(train), 'target':torch.from_numpy(target)}

        return data