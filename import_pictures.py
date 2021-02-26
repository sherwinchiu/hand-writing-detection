import os
import cv2
"""
# import_pictures.py
# Sherwin Chiu and Vivian Dai
# 2021/02/26
# imports the images and... saves it.. somewhere...
"""
#----------------------------CONSTANTS----------------------------#
NUMBER_OF_DIGITS_IN_DATA = 3

#----------------------VARIABLE DECLARATIONS----------------------#
training_images = []
training_labels = []

#----------------------------FUNCTIONS----------------------------#
def reformat(name):
    '''Takes in a string name, returns the name reformatted 
    (without the numbers and image file type declaration'''
    ind = len(name)
    for i in range(len(name) - 1, 0, -1):
        if name[i] == ".":
            ind = i
            return name[:i - NUMBER_OF_DIGITS_IN_DATA]
    
#-------------------------------MAIN------------------------------#

for filename in os.listdir("./training_assets"):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        training_images.append(cv2.imread(f"training_assets/{filename}"))
        training_labels.append(reformat(filename))


print(training_images, training_labels)