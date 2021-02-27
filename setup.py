import os
import cv2
import identify
from tensorflow.keras import datasets, layers, models
from skimage.transform import resize
from random import randint
"""
# setup.py
# Sherwin Chiu and Vivian Dai
# 2021/02/26
# imports the images and creates a trained .model file
"""
#----------------------------CONSTANTS----------------------------#
NUMBER_OF_DIGITS_IN_DATA = 3

#----------------------VARIABLE DECLARATIONS----------------------#
training_images = []
training_labels = []
testing_images = []
testing_labels = []

#----------------------------FUNCTIONS----------------------------#
def reformat(name):
    '''Takes in a string name, returns the name reformatted 
    (without the numbers and image file type declaration'''
    split = name.split()
    if(len(split) == 2):
        return split[0]
    for i in range(len(name) - 1, 0, -1):
        if name[i] == ".":
            return name[:i - NUMBER_OF_DIGITS_IN_DATA]
    
def getRandomData(n, image_list, label_list):
    a, b = []
    for i in range(n):
        r = randint(0, len(image_list) - 1)
        a.append(image_list[r])
        b.append(label_list[r])
    return a, b
#-------------------------------MAIN------------------------------#
# imports images
for filename in os.listdir("./training_assets"):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image = cv2.imread(f"training_assets/{filename}")
        image = resize(image, (32, 32, 3))
        image = image/255
        training_images.append(image)
        training_labels.append(identify.classification.index(reformat(filename)))

print("Import complete")
testing_images, testing_labels = getRandomData(100, training_images, training_labels)
# training
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (32, 32, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(10, activation = "softmax"))

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(training_images, training_labels, epochs = 10, validation_data = (testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}\nAccuracy: {accuracy}")

model.save("handwriting.model")