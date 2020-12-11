# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:34:45 2020

@author: Ryan and Deric
"""
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.datasets import cifar10

#Load images from the location with our code
for i in range(1, 21):
    img = load_img('image{}.jpg'.format(i), target_size=(32, 32))
    plt.savefig('imageOut{}.jpg'.format(i + 20))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    np.save('image{}.file'.format(i), img)
    
    
#Load Dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#create the classes list such as section 1
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Create Five Subplots similar to Section 1
f, axarr = plt.subplots(1, 5)
f.set_size_inches(16, 6)

X_train[0] = np.load('image9.file.npy')
X_train[1] = np.load('image12.file.npy')
X_train[2] = np.load('image16.file.npy')
X_train[3] = np.load('image19.file.npy')
X_train[4] = np.load('image20.file.npy')
for i in range(5):
    img = X_train[i]
    axarr[i].imshow(img)
plt.show()

#Replace corresponding labels using the cifar_classes number above
y_train[0] = 4
y_train[1] = 5
y_train[2] = 8
y_train[3] = 9
y_train[4] = 9

print('Our Training images and their labels: ' + str([x[0] for x in y_train[0:5]]))
print('Our classes for the labels: ' + str([cifar_classes[x[0]] for x in y_train[0:5]]))