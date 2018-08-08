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

class1center=[-3, -3]
class2center=[-3, 3]
class3center=[3, -3]
class4center=[3, 3]
class5center=[0, 0]


points1=class1center+.4*np.random.randn(numsamples, 2)
points2=class2center+np.random.randn(numsamples, 2)
points3=class3center+2*np.random.randn(numsamples, 2)
points4=class4center+np.random.randn(numsamples, 2)
points5=class5center+.5*np.random.randn(numsamples, 2)

plt.ion()
fig=plt.figure()
ax=fig.add_subplot(111)#, projection='3d')

ax.scatter(points1[:, 0], points1[:, 1], c='r')
ax.scatter(points2[:, 0], points2[:, 1], c='b')
ax.scatter(points3[:, 0], points3[:, 1], c='g')
ax.scatter(points4[:, 0], points4[:, 1], c='m')
ax.scatter(points5[:, 0], points5[:, 1], c='c')

plt.show()
plt.pause(0.01)

data=np.vstack((points1, points2, points3, points4, points5))
labels=np.vstack((np.zeros((numsamples, 1)), np.ones((numsamples, 1)), 2*np.ones((numsamples, 1)), 3*np.ones((numsamples, 1)), 4*np.ones((numsamples, 1))))
labels=tensorflow.keras.utils.to_categorical(labels, num_classes=5)
model=Sequential()
model.add(Dense(10, activation='relu', input_dim=2))
model.add(Dense(5, activation='softmax'))
print(model.summary())
model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
test_points=[np.array([-5+i*.2, -5+j*.2]) for i in range(0, 50) for j in range(0, 50)]
for i in range(0, 25):

    test_outs=model.predict(np.array(test_points))
    test1=np.array((0, 2))
    test2=np.array((0, 2))
    test3=np.array((0, 2))
    test4=np.array((0, 2))
    test5=np.array((0, 2))
    for i in range(0, len(test_outs)):
        label=np.argmax(test_outs[i])
        if label==0:
            test1=np.vstack((test1, test_points[i]))
        elif label==1:
            test2=np.vstack((test2, test_points[i]))
        elif label==2:
            test3=np.vstack((test3, test_points[i]))
        elif label==3:
            test4=np.vstack((test4, test_points[i]))
        else:
            test5=np.vstack((test5, test_points[i]))

    if test1.shape[0]>0 and test2.shape[0]>0 and test3.shape[0]>0 and test4.shape[0]>0 and test5.shape[0]>0:
        ax.clear()
        ax.scatter(test1[:, 0], test1[:, 1], c='r', marker='o', linewidths=0.0)
        ax.scatter(test2[:, 0], test2[:, 1], c='b', marker='o', linewidths=0.0)
        ax.scatter(test3[:, 0], test3[:, 1], c='g', marker='o', linewidths=0.0)
        ax.scatter(test4[:, 0], test4[:, 1], c='m', marker='o', linewidths=0.0)
        ax.scatter(test5[:, 0], test5[:, 1], c='c', marker='o', linewidths=0.0)
        ax.scatter(points1[0:100, 0], points1[0:100, 1], c='r')
        ax.scatter(points2[0:100, 0], points2[0:100, 1], c='b')
        ax.scatter(points3[0:100, 0], points3[0:100, 1], c='g')
        ax.scatter(points4[0:100, 0], points4[0:100, 1], c='m')
        ax.scatter(points5[0:100, 0], points5[0:100, 1], c='c')
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
