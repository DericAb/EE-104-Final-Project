# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:34:45 2020

@author: Ryan and Deric
"""
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import matplotlib.pyplot as plt
 
# load and prepare the image
def load_image(filename):
	# load the image
    img = load_img(filename, target_size=(32, 32))
    plt.imshow(img)
	# convert to array
    img = img_to_array(img)
	# reshape into a single sample with 3 channels
    img = img.reshape(1, 32, 32, 3)
	# prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img
 
# load an image and predict the class
def run_example():
    # load the image
    img = load_image('image17.jpg')
    # load model
    model = load_model('FINAL.h5')
    # predict the class
    result = model.predict_classes(img)
    print("\n")
    print(result[0])
 
# entry point, run the example
run_example()

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(cifar_classes)