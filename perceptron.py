import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import tensorflow
import tensorflow.keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

class1center=[3, 3]
class2center=[0, 0]

numsamples=2000

points1=class1center+np.random.randn(numsamples, 2)
points2=class2center+.5*np.random.randn(numsamples, 2)

plt.ion()
fig=plt.figure()
ax=fig.add_subplot(111)#, projection='3d')

#ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='r')
#ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='b')
ax.scatter(points1[:, 0], points1[:, 1], c='r')
ax.scatter(points2[:, 0], points2[:, 1], c='b')

plt.show()
plt.pause(0.01)

data=np.vstack((points1, points2))
labels=np.vstack((np.ones((numsamples, 1)), np.zeros((numsamples, 1))))
model=Sequential()
model.add(Dense(1, input_dim=2))
model.add(Activation('sigmoid'))
print(model.summary())
model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
for i in range(0, 50):
    w=model.get_weights()
    a=w[0][0][0]
    b=w[0][1][0]
    c=w[1][0]
    #fig=plt.figure()
    #ax=fig.add_subplot(111)#, projection='3d')
    #ax.scatter(points1[:, 0], points1[:, 1], c='r')
    #ax.scatter(points2[:, 0], points2[:, 1], c='b')
    #ax+by+c=0 ---> y=-(a/b)x-c/b
    x=np.array([-1, 4])
    y=-(a/b)*x-c/b

    ax.clear()
    ax.scatter(points1[:, 0], points1[:, 1], c='r')
    ax.scatter(points2[:, 0], points2[:, 1], c='b')
    ax.plot(x, y, 'g-')
    plt.pause(0.01)
    #plt.show()
    
    model.fit(data, labels, epochs=1, batch_size=32)
