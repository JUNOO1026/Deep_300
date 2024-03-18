import os
import glob
import shutil

train_path = 'C:/Users/jun/Downloads/dataset/training/'
test_path = 'C:/Users/jun/Downloads/dataset/testing/'



if not os.path.exists(os.path.join(train_path, 'train_images')):
    os.makedirs(os.path.join(train_path, 'train_images'))
if not os.path.exists(os.path.join(train_path, 'train_matte')):
    os.makedirs(os.path.join(train_path, 'train_matte'))
if not os.path.exists(os.path.join(test_path, 'test_images')):
    os.makedirs(os.path.join(test_path, 'test_images'))
if not os.path.exists(os.path.join(test_path + 'test_matte')):
    os.makedirs(os.path.join(test_path, 'test_matte'))

final_train_images = os.path.join(train_path, 'train_images')
final_train_mattes = os.path.join(train_path, 'train_matte')
final_test_images = os.path.join(test_path, 'test_images')
final_test_matte = os.path.join(test_path, 'test_matte')

print(final_train_images)
print(final_train_mattes)
keyword = '_matte'

for train_images in os.listdir(train_path):
    fp = os.path.join(train_path, train_images)
    if os.path.isfile(fp) and keyword in train_images:
        shutil.move(fp, final_train_mattes)
    else:
        shutil.move(fp, final_train_images)

for test_images in os.listdir(test_path):
    fp = os.path.join(test_path, test_images)

    if os.path.isfile(fp) and keyword in test_images:
        shutil.move(fp, final_test_matte)
    else:
        shutil.move(fp, final_test_images)


