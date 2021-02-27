# Imports
import cv2
from identify import classification
import tensorflow as tf
from tensorflow.keras import models
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import cifar10
from skimage.transform import resize
"""
# main.py
# Sherwin Chiu and Vivian Dai
# 2021/02/26
# Allows user to input the path to an image and identifies the letter
"""

# Style of pyplot
plt.style.use('fivethirtyeight')

# Loading data

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
model = models.load_model("handwriting.model")
path = input("give path to image")
img = cv2.imread(path)
img = resize(img, (32, 32, 3))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)
print(classification[index])