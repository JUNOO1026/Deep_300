import os
import cv2
import torch
import numpy as np

from augmentation import one_hot_encode

BASE_PATH = 'C:/Users/jun/Downloads/dataset/'
dir_train_img = os.path.join(BASE_PATH, 'train_images')
dir_train_lbl = os.path.join(BASE_PATH, 'train_labels')
dir_val_img = os.path.join(BASE_PATH, 'val_images')
dir_val_lbl = os.path.join(BASE_PATH, 'val_labels')


class_names = ['background', 'person']
class_rgb_values = [[0, 0, 0], [255, 255, 255]]


select_class_indices = [class_names.index(cls.lower()) for cls in class_names]
select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, class_rgb_values=None, transforms=None):
        self.img_dir = [os.path.join(image_dir, image_idx) for image_idx in sorted(os.listdir(image_dir))]
        self.lbl_dir = [os.path.join(label_dir, label_idx) for label_idx in sorted(os.listdir(label_dir))]
        self.class_rgb_values = class_rgb_values
        self.transforms = transforms

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.img_dir[idx]), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread(self.lbl_dir[idx]), cv2.COLOR_BGR2RGB)

        label = one_hot_encode(label, self.class_rgb_values).astype('float32')

        if self.transforms:
            image = image.transpose(2, 0, 1).astype('float32')
            label = label.transpose(2, 0, 1).astype('float32')

        return image, label


dataset = FaceDataset(dir_train_img, dir_train_lbl, class_rgb_values=select_class_rgb_values)
