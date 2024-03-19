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
        print(train_img_fp)

        plt.imshow(np.array(Image.open(train_img_fp)))
        ## 내일 해당 부분부터 다시 짜야 함.
        ## AttributeError: 'Dataset' object has no attribute 'train_matte_path'. Did you mean: 'train_img_path'?
        # 이와 같은 에러가 발생했으므로 해결을 해야함.
        train_img, train_matte = self.train_img_dataset(train_img_fp)
        train_img = np.array(train_img)
        train_matte = np.array(train_matte)
        print(train_img.shape, train_matte.shape)
        # train_img = train_img / 255.0
        # train_matte = train_matte / 255.0
        print(type(train_img))
        train_img = train_img.astype(np.float32)
        train_matte = train_matte.astype(np.float32)

        if train_img.ndim == 2:
            train_img = train_img[:, :, np.newaxis]
        if train_matte.ndim == 2:
            train_matte = train_matte[:, :, np.newaxis]

        # data = {"train":train_img, "target":train_matte}
        # print(data['train'].shape)
        # print(data['target'].shape)
        if self.transform:
            data = self.transform(train_img)

        return data

    def train_img_dataset(self, train_img_path):
        images = []
        mattes = []
        train_images = []
        train_mattes = []

        for file in os.listdir(train_img_path):
            if '_matte' in file:
                mattes.append(file)
            else:
                images.append(file)

        for idx, file in enumerate(images):
            train_images.append(np.array(Image.open(os.path.join(train_img_path, images[idx]))))
        for idx, file in enumerate(mattes):
            train_mattes.append(np.array(Image.open(os.path.join(train_img_path, mattes[idx]))))
        return train_images, train_mattes

    def transform(self, image):
        transforms_ops = transforms.Compose([
            transforms.ToTensor()
        ])
        return transforms_ops(image)
#
# class ToTensor(object):
#     def __call__(self, data):
#         train, target = data['train'], data['target']
#
#         print(train.shape)
#         print(target.shape)
#
#         train = train.transpose((2, 0, 1)).astype(np.float32)
#         target = target.transpose((2, 0, 1)).astype(np.float32)
#
#         data = {'train':torch.from_numpy(train), 'target':torch.from_numpy(target)}
#
#         return data