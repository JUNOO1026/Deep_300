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
%matplotlib inline

SEED = 34

mnist = keras.datasets.mnist
((train_images, train_labels), (test_images, test_labels))= mnist.load_data()

print(f'train_images : {train_images.shape}')
print(f'train_labels : {train_labels.shape}')
print(f'test_images : {test_images.shape}')
print(f'test_labels : {test_labels.shape}')

train_images[0].shape

plt.figure(figsize=(5, 5))

plt.imshow(train_images[0])
plt.colorbar()
plt.show()
print(train_labels[0])

print(list(filter(lambda x : x != 0, train_images[0].reshape(-1))))

print(train_images.dtype)
print(train_labels.dtype)
print(test_images.dtype)
print(test_labels.dtype)

print(type(train_images))
print(type(train_labels))
print(type(test_images))
print(type(test_labels))

print(train_images.reshape(-1))
print(train_images.shape)
print(min(train_images.reshape(-1)), max(train_images.reshape(-1)))
print(min(test_images.reshape(-1)), max(test_labels.reshape(-1)))

train_images.astype(np.float64)
test_images.astyper(np.float64)

print((train_images / 255).reshape(-1))

train_images = train_images / 255.0
test_images = test_images / 255.0

print(list(filter(lambda x : x != 0, train_images[0].reshape(-1)))[:10])
print(list(filter(lambda x : x != 0, train_labels.reshape(-1)))[:10])
print(list(filter(lambda x : x != 0, test_images[0].reshape(-1)))[:10])
print(list(filter(lambda x : x != 0, test_labels.reshape(-1)))[:10])

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
print(train_images.dtype, train_labels.dtype, test_images.dtype, test_labels.dtype)