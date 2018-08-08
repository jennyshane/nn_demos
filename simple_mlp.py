import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import tensorflow
import tensorflow.keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

numsamples=2000

class1center=[0, 0]

class2param=np.random.rand(numsamples)*np.pi*1.9
class2center=np.array([4*np.sin(class2param), 4*np.cos(class2param)]).T

points1=class1center+np.random.randn(numsamples, 2)
points2=class2center+.4*np.random.randn(numsamples, 2)

plt.ion()
fig=plt.figure()
ax=fig.add_subplot(111)#, projection='3d')

ax.scatter(points1[:, 0], points1[:, 1], c='r')
ax.scatter(points2[:, 0], points2[:, 1], c='b')

plt.show()
plt.pause(0.01)

data=np.vstack((points1, points2))
labels=np.vstack((np.ones((numsamples, 1)), np.zeros((numsamples, 1))))
model=Sequential()
model.add(Dense(10, activation='relu', input_dim=2))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
test_points=[np.array([-5+i*.2, -5+j*.2]) for i in range(0, 50) for j in range(0, 50)]
for i in range(0, 25):

    test_outs=model.predict(np.array(test_points))
    test1=np.array((0, 2))
    test2=np.array((0, 2))
    for i in range(0, len(test_outs)):
        if test_outs[i]>.5:
            test1=np.vstack((test1, test_points[i]))
        else:
            test2=np.vstack((test2, test_points[i]))


    ax.clear()
    ax.scatter(test1[:, 0], test1[:, 1], c='r', marker='o', linewidths=0.0)
    ax.scatter(test2[:, 0], test2[:, 1], c='b', marker='o', linewidths=0.0)
    ax.scatter(points1[0:100, 0], points1[0:100, 1], c='r')
    ax.scatter(points2[0:100, 0], points2[0:100, 1], c='b')
    plt.pause(0.01)
    
    model.fit(data, labels, epochs=1, batch_size=32)

plt.ioff()

fig=plt.figure()
ax2=fig.add_subplot(111)#, projection='3d')
ax2.scatter(test1[:, 0], test1[:, 1], c='r', marker='o', linewidths=0.0)
ax2.scatter(test2[:, 0], test2[:, 1], c='b', marker='o', linewidths=0.0)
ax2.scatter(points1[0:100, 0], points1[0:100, 1], c='r')
ax2.scatter(points2[0:100, 0], points2[0:100, 1], c='b')

plt.show()

