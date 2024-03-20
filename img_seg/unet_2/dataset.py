import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FaceDataset(Dataset):
    def __init__(self, images_dir, matte_dir, augmentation=None, preprocessing=None):

        self.img_path = [os.path.join(images_dir, image_idx) for image_idx in sorted(os.listdir(images_dir))]
        self.mat_path = [os.path.join(matte_dir, matte_idx) for matte_idx in sorted(os.listdir(matte_dir))]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.img_path[idx]), cv2.COLOR_BGR2RGB)
        matte = cv2.cvtColor(cv2.imread(self.mat_path[idx]), cv2.COLOR_BGR2RGB)

        if self.augmentation:
            sample = self.augmentation(image=image, matte=matte)
            image, matte = sample['image'], sample['matte']

        if self.preprocessing:
            sample = self.preprocessing(image=image, matte=matte)
            image, matte = sample['image'], sample['matte']

        return image, matte
