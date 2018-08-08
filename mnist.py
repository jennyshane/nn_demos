import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import tensorflow
import tensorflow.keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Embedding, Input 
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(x_train.dtype)
print(x_test.dtype)

num_classes=10
test_labs=to_categorical(y_test)
train_labs=to_categorical(y_train)
test_imgs=x_test.reshape(x_test.shape[0], 28, 28, 1)
train_imgs=x_train.reshape(x_train.shape[0], 28, 28, 1)
test_imgs=test_imgs.astype('float32')
train_imgs=train_imgs.astype('float32')
test_imgs/=255
train_imgs/=255

model=Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(train_imgs, train_labs, batch_size=128, epochs=40, verbose=1, validation_data=(test_imgs, test_labs), shuffle=True)
score=model.evaluate(test_imgs, test_labs, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
