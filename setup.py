import os
import cv2
import identify
import matplotlib.pyplot as plt 
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
DIRECTORIES = ['./training-data/a', './training-data/A_0', './training-data/b', './training-data/B_0', './training-data/c', './training-data/C_0', './training-data/d', './training-data/D_0', './training-data/e', './training-data/E_0', './training-data/f', './training-data/F_0', './training-data/g', './training-data/G_0', './training-data/h', './training-data/H_0', './training-data/i', './training-data/I_0', './training-data/j', './training-data/J_0', './training-data/k', './training-data/K_0', './training-data/l', './training-data/L_0', './training-data/m', './training-data/M_0', './training-data/n', './training-data/N_0', './training-data/o', './training-data/O_0', './training-data/p', './training-data/P_0', './training-data/q', './training-data/Q_0', './training-data/r', './training-data/R_0', './training-data/s', './training-data/S_0', './training-data/t', './training-data/T_0', './training-data/u', './training-data/U_0', './training-data/v', './training-data/V_0', './training-data/w', './training-data/W_0', './training-data/x', './training-data/X_0', './training-data/y', './training-data/Y_0', './training-data/z', './training-data/Z_0']

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
    images = []
    labels = []
    for i in range(n):
        r = randint(0, len(image_list) - 1)
        images.append(image_list[r])
        labels.append(label_list[r])
    return images, labels
#-------------------------------MAIN------------------------------#
# imports images
for directory in DIRECTORIES:
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image = cv2.imread(f"{directory}/{filename}", cv2.IMREAD_GRAYSCALE)
            if image is not None:
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

hist = model.fit(training_images, training_labels, epochs = 10, validation_data = (testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}\nAccuracy: {accuracy}")

plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Val"], loc = "upper right")
plt.show()

model.save("handwriting.model")