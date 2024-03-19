import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 파일이 있는 폴더 경로를 설정합니다.
folder_path = 'C:/Users/jun/Downloads/dataset/training/'

images = []
mattes = []

train_images = []
train_mattes = []

train_images = 'C:/Users/jun/Downloads/dataset/training/00249.png'

a = Image.open(train_images)

b = a.size
c = a.n_frames

print(b)
print(c)

# for file in os.listdir(folder_path):
#     if '_matte' in file:
#         mattes.append(file)
#     else:
#         images.append(file)

# a = np.array(Image.open(os.path.join(folder_path, images[0])))
# print(a.shape)
# print(type(a))
# plt.imshow(a)
# plt.show()


# print(images)
#
# for idx, file in enumerate(images):
#     train_images.append(np.array(Image.open(os.path.join(folder_path, images[idx]))))
# for idx, file in enumerate(mattes):
#     train_mattes.append(np.array(Image.open(os.path.join(folder_path, mattes[idx]))))
#
# print(np.array(train_images).shape, np.array(train_mattes)[..., np.newaxis].shape)

# train_img = np.array(Image.open(os.path.join(folder_path, train_images[0])))
# print(train_img.shape)
# print(train_img.dtype)
# print(type(train_img))
# plt.imshow(train_img)
# plt.show()