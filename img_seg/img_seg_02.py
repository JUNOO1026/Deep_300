import torch
import torch.nn as nn
import glob
from PIL import Image
import numpy as np
from skimage import color
import matplotlib.pyplot as plt

train_path = 'C:/Users/jun/Downloads/dataset/dataset/training/'
test_path = 'C:/Users/jun/Downloads/dataset/dataset/testing/'


train_file_list = glob.glob(train_path + '*.png')
test_file_list = glob.glob(test_path + '*.png')

train_images = []
train_mattes = []
test_images = []
test_mattes = []

for file_path in train_file_list:
    if '_matte' in file_path:
        train_mattes.append(np.array(Image.open(file_path)))
    else:
        train_images.append(np.array(Image.open(file_path)))

for file_path in test_file_list:
    if '_matte' in file_path:
        test_mattes.append(np.array(Image.open(file_path)))
    else:
        test_images.append(np.array(Image.open(file_path)))



train_images = np.array(train_images)
train_mattes = np.array(train_mattes)
test_images = np.array(test_images)
test_mattes = np.array(test_mattes)

print(f"train_image.shape : {train_images.shape}")
print(f"train_mattes.shape : {train_mattes.shape}")
print(f"test_images.shape : {test_images.shape}")
print(f"test_mattes.shape : {test_mattes.shape}")


print(train_images.max(), train_images.min())
print(train_mattes.max(), train_mattes.min())


gray_train_images = np.array([color.rgb2gray(img).reshape(800, 600, 1) for img in train_images])
gray_test_images = np.array([color.rgb2gray(img).reshape(800, 600, 1) for img in test_images])

print(f"gray_train_images.shape : {gray_train_images.shape}")
print(f"gray_test_images.shape : {gray_test_images.shape}")



class ConvBlock(nn.Module):
    def __init__(self, in_channel, expansion):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channel, 64 * expansion, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(64*expansion),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channel, 64 * expansion, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(64 * expansion),
                                  nn.ReLU()
                                  )

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channel, **kwargs):

        self.conv




summary(model, input_size=(1700, 3, 800, 600), device='cuda')