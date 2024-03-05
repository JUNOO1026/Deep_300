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

# 이미지
plt.figure()
plt.imshow(np.random.random((28, 28)), cmap='gray')
plt.colorbar()
plt.show()

plt.imshow(np.random.normal(0.0, 0.1, (28, 28)), cmap='gray')
plt.colorbar()
plt.show()

noisy_image = train_images[5] + np.random.normal(0.5, 0.1, (28, 28))

plt.imshow(noisy_image,  cmap='gray')
plt.show()


noisy_image[noisy_image > 1] = 1.0
noisy_image[noisy_image > 1]

plt.imshow(noisy_image, cmap='gray')
plt.colorbar()
plt.show()

print(train_images.shape[0])
print(test_images.shape)

np.random.normal(0.0, 0.1, (28, 28))

train_noisy_images = train_images + np.random.normal(0.0, 0.1, train_images.shape)
train_noisy_images[train_noisy_images > 1.0] = 1.0
test_noisy_images = test_images + np.random.normal(0.0, 0.1, test_images.shape)
test_noisy_images[test_noisy_images > 1.0] = 1.0
print(train_noisy_images.shape)

plt.imshow(train_noisy_images[0])
plt.colorbar()
plt.show()

plt.imshow(test_noisy_images[0])
plt.colorbar()
plt.show()

#perfect
plt.imshow(train_noisy_images[:5].transpose(1, 0, 2).reshape(28, -1), cmap='gray')
plt.colorbar()
plt.show()
# false
plt.imshow(train_noisy_images[:5].reshape(28, -1), cmap='gray')
plt.colorbar()
plt.show()

train_noisy_images[:5].reshape(28, -1).shape

print(train_noisy_images[:5].reshape(28, -1))

plt.imshow(train_noisy_images[:5].transpose(1, 0, 2).reshape(28, -1), cmap='gray')
plt.colorbar()
plt.show()

plt.imshow(train_noisy_images[:2].transpose(1, 0, 2).reshape(28, -1), cmap='gray')
plt.imshow(train_noisy_images[:2].transpose(1, 0, 2).reshape(28, -1), cmap='gray')

from keras.utils import to_categorical

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

print(train_labels.shape)
print(test_labels.shape)

from keras.layers import SimpleRNN
from keras.layers import Dense, Input
from keras.models import Model

inputs = Input(shape=(28, 28))
x1 = SimpleRNN(64, activation='tanh')(inputs)
x2 = Dense(10, activation='softmax')(x1)
model = Model(inputs, x2)


model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(train_noisy_images, train_labels, validation_data=(test_noisy_images, test_labels), epochs=5, verbose=2)

plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_accuracy'], label='val_accuracy')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend(loc='upper left')
plt.show()

# test_noisy_images[0]

res = model.predict(test_noisy_images[:10])

plt.imshow(np.concatenate([test_noisy_images[1], test_images[1]], axis=1))
plt.show()

res[0].argmax()
plt.bar(range(10), res[1], color='red')
plt.bar(np.array(range(10)) + 0.5, test_labels[1])
plt.show()