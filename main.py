# Imports
import tensorflow as tf
from tensorflow import keras, models
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import cifar10


# Style of pyplot
plt.style.use('fivethirtyeight')

# Loading data

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
model = models.load_model("handwriting.model")
# TODO: predict letter