# hand-writting-detection

[![made-for-VSCode](https://img.shields.io/badge/Made%20for-VSCode-1f425f.svg)](https://code.visualstudio.com/)

[![python](https://img.shields.io/badge/python-v3.6%2B-blue)](https://www.python.org/downloads/)

## Description

Looks at two different signature samples, and tells whether or not the handwriting is a forgery or not

## Usage

Can fill the `training-data` folder with whatever letter images as long as the names are in the correct format, either:

1. `[letter]xxx.filetype` where `[letter]` represents which letter it's supposed to be, `xxx` are 3 distinct digits, and the filetype is either [.png](https://fileinfo.com/extension/png) or [.jpg](https://fileinfo.com/extension/jpg)
2. `[letter] (X).filetype` where `[letter]` represents the letter it's supposed to be, `X` is any number, and filetype is either [.png](https://fileinfo.com/extension/png) or [.jpg](https://fileinfo.com/extension/jpg)

## Dependancies

* [tensorflow](https://pypi.org/project/tensorflow/): machine learning library
* [keras](https://pypi.org/project/Keras/): deep learning, nerual network APIs
* [sci-kit image](https://pypi.org/project/scikit-image/): image processing (used to resize image)
* [numpy](https://pypi.org/project/numpy/): array processing (images are stored in an array)
* [matplotlib](https://pypi.org/project/matplotlib/): plotting graphs

Have fun!

![Kangaroo](https://github.com/sherwinchiu/hand-writing-detection/blob/master/training-data/K_0/K%20(2).png)
