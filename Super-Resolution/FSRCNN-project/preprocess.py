import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpim

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


def extract_patches(img, req_size, stride): # 패치 추출
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
        patches_HR = extract_patches(img, req_size, stride) # 고해상도 패치 추출
        for patch_HR in patches_HR:
            patch_LR = cv2.resize(patch_HR, (inp_size, inp_size)) # 고해상도 패치 사이즈를 줄여서 저해상도패치 추출
            inputs.append(patch_LR) # 저해상도 패치 매핑
            labels.append(patch_HR) # 고해상도 패치 매핑

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
    
    # inputs,labels = flip_randomly(inputs,labels)

    return inputs, labels

def flip_randomly(lowres_img, highres_img):
    """이미지를 무작위로 좌우로 뒤집는다."""

    # 0부터 1까지의 균일 분포에서 무작위 숫자를 생성
    rn = tf.random.uniform(shape=(), maxval=1)
    # 만약 rn이 0.5보다 작으면 원본 이미지를 반환하고, 그렇지 않으면 뒤집힌 이미지를 반환
    return tf.cond(
        rn < 0.5,
        lambda: (lowres_img, highres_img),
        lambda: (
            tf.image.flip_left_right(lowres_img),
            tf.image.flip_left_right(highres_img),
        ),
    )
