import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import glob
from IPython.display import Image


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


from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model
from keras.layers import BatchNormalization, Dropout, Activation, MaxPool2D, concatenate

def conv2d_block(x, channel):
    x = Conv2D(channel, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(channel, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def unet():
    inputs = Input((100, 75, 3))
    c1 = conv2d_block(inputs, 16)
    p1 = MaxPool2D((2, 2))(c1)
    p1 = Dropout(0.1)(p1)

    c2 = conv2d_block(p1, 32)
    p2 = MaxPool2D((2, 2))(c2)
    p2 = Dropout(0.1)(p2)

    c3 = conv2d_block(p2, 64)
    p3 = MaxPool2D((2, 2))(c3)
    p3 = Dropout(0.1)(p3)

    c4 = conv2d_block(p3, 128)
    p4 = MaxPool2D((2, 2))(c4)
    p4 = Dropout(0.1)(p4)

    c5 = conv2d_block(p4, 256)

    u6 = Conv2DTranspose(128, 2, 2, output_padding=(0, 1))(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(0.1)(u6)
    c6 = conv2d_block(u6, 128)

    u7 = Conv2DTranspose(64, 2, 2, output_padding=(1, 0))(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(0.1)(u7)
    c7 = conv2d_block(u7, 64)

    u8 = Conv2DTranspose(32, 2, 2, output_padding=(0, 1))(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(0.1)(u8)
    c8 = conv2d_block(u8, 32)

    u9 = Conv2DTranspose(16, 2, 2, output_padding=(0, 1))(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(0.1)(u9)
    c9 = conv2d_block(u9, 16)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = Model((inputs, outputs))

    return model

# output_padding에 대한 고찰

# output_padding을 간단히 말하자면 필요한 부분에 픽셀을 넣는 것이다.
# u7에서 output_padding을 적용하지 않으면 concatenate를 할때 문제가 발생한다.
# u7.shape = (24, 18, 64)
# c3.shape = (25, 18, 64)
# 이 상태에서 u7과 c3를  concatenate를 하게 되면 오류가 발생.

# 따라서 Height에 1pixel을 추가하여 오류를 방지한다.
