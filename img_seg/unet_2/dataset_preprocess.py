import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil

BASE_PATH = 'C:/Users/jun/Downloads/dataset/'
TRAIN_PATH = 'C:/Users/jun/Downloads/dataset/training/'
TEST_PATH = 'C:/Users/jun/Downloads/dataset/testing/'

dir_train_img = os.path.join(BASE_PATH, 'train_images')
dir_train_lbl = os.path.join(BASE_PATH, 'train_labels')
dir_test_img = os.path.join(BASE_PATH, 'test_images')
dir_test_lbl = os.path.join(BASE_PATH, 'test_labels')

#
image_paths = [os.path.join(dir_train_img, image_id) for image_id in sorted(os.listdir(dir_train_img))]
a = np.array(Image.open(image_paths[0]))

plt.imshow(a)
plt.show()



#
# for image_id in sorted(os.listdir(dir_train_img)):
#     print(image_id)
#
# for matte_id in sorted(os.listdir(dir_train_lbl)):
#     print(matte_id)

#
# if not os.path.exists(dir_train_img):
#     os.makedirs(dir_train_img)
# if not os.path.exists(dir_train_lbl):
#     os.makedirs(dir_train_lbl)
# if not os.path.exists(dir_test_img):
#     os.makedirs(dir_test_img)
# if not os.path.exists(dir_test_lbl):
#     os.makedirs(dir_test_lbl)
#
#
# for file in os.listdir(TRAIN_PATH):
#     if '_matte' in file:
#         shutil.move(TRAIN_PATH + file, dir_train_lbl)
#     else:
#         shutil.move(TRAIN_PATH + file, dir_train_img)
#
# for file in os.listdir(TEST_PATH):
#     if '_matte' in file:
#         shutil.move(TEST_PATH + file, dir_test_lbl)
#     else:
#         shutil.move(TEST_PATH + file, dir_test_img)


# for file in os.listdir(dir_train_lbl):
#     img = [file for file in os.listdir(dir_train_lbl) if '_matte' in file]
#
#
# print(img)
