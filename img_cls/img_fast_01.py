import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import Image

warnings.filterwarnings('ignore')
import glob
from PIL import Image
import numpy as np

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

from skimage import color
train_mattes = np.array([color.gray2rgb(img) for img in train_mattes])
test_mattes = np.array([color.gray2rgb(img) for img in test_mattes])


from keras.layers import Dense, Input, Conv2D, UpSampling2D, Flatten, Reshape
from keras.models import Model

def AE():
    inputs = Input((100, 75, 3))
    x = Conv2D(32, 3, 2, activation="relu", padding='same')(inputs)
    x = Conv2D(64, 3, 2, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, 2, activation='relu', padding='same')(x)
    x = Flatten()(x)
    latent = Dense(10)(x)

    x = Dense((13 * 10 * 128))(latent)
    x = Reshape((13, 10, 128))(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (2, 2), (1, 1), activation='relu', padding='valid')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (1, 1), (1, 1), activation='relu', padding='valid')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32, (1, 2), (1, 1), activation='relu', padding='valid')(x)

    x = Conv2D(1, (1,1), (1,1), activation='sigmoid')(x)

    model = Model(inputs, x)
    return model


model = AE()

model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
hist = model.fit(train_images, train_mattes, validation_data = (test_images, test_mattes), epochs=100, verbose=1 )

plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_accuracy'], label='val_accuracy')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend(loc='center right')
plt.show()

res = model.predict(test_images[0:1])
print(res.reshape(100, 75, 1).shape, test_mattes[0].reshape(100, 75, 1).shape)
pred = np.concatenate([(res[0] > 0.5).astype(np.float64), test_mattes[0]]).reshape((2, 100, 75, 1)).transpose([1, 0, 2, 3]).reshape((100, -1))
# pred1 = np.concatenate([res[0], test_mattes[0]]).reshape((2, -1, 75, 1)).transpose([1, 0, 2, 3]).reshape((100, -1))
plt.imshow(pred, cmap='gray')
plt.show()

five_img = (model.predict(test_images[:5]) > 0.5).astype(np.float64)

pred = np.concatenate([five_img, test_mattes[:5]], axis=2).transpose([1, 0, 2, 3]).reshape((100, -1))
print(np.concatenate([five_img, test_mattes[:5]], axis=2).transpose([1, 0, 2, 3]).shape)

plt.imshow(pred, cmap='gray')
plt.show()