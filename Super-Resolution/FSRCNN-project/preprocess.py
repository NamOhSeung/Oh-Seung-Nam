import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, mean_squared_error
from tensorflow.keras.preprocessing import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

# import albumentations as A
from constants import *
import cv2
import os
from glob import glob
import random

img_path1 = "./image_data/yang91"
img_path2 = "./image_data/yang91"

# 전처리 과정 가져오기

scale = 2
req_size = 20
inp_size = int(req_size/scale)
stride = 15

def PSNR(Original,compressed):
    mse = np.mean((Original-compressed)**2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    pixel_max = np.max(compressed).astype("uint8")
    PSNR = 20*np.log10(pixel_max / np.sqrt(mse))
    return PSNR

def extract_patches(img, req_size, stride):
    patches = []
    shapes = img.shape
    for i in range(0, shapes[0] - req_size + 1, stride):
        for j in range(0, shapes[1] - req_size + 1, stride):
            patch = img[i:i + req_size, j:j + req_size]
            patches.append(patch)
    return patches

def preprocessing(img_path1, img_path2, req_size, inp_size, stride):
    inputs = []
    labels = []

    for file_path in os.listdir(img_path1):
        full_path = os.path.join(img_path1, file_path)
        img = cv2.imread(full_path)
        img = np.asarray(img)
        patches_HR = extract_patches(img, req_size, stride)
        for patch_HR in patches_HR:
            patch_LR = cv2.resize(patch_HR, (inp_size, inp_size))
            inputs.append(patch_LR)
            labels.append(patch_HR)

    count = 0
    for file_path in os.listdir(img_path2):
        count += 1
        if count == 300:
            break
        full_path = os.path.join(img_path2, file_path)
        img = cv2.imread(full_path)
        img = np.asarray(img)
        patches_HR = extract_patches(img, req_size, stride)
        for patch_HR in patches_HR:
            patch_LR = cv2.resize(patch_HR, (inp_size, inp_size))
            inputs.append(patch_LR)
            labels.append(patch_HR)

    return inputs, labels
