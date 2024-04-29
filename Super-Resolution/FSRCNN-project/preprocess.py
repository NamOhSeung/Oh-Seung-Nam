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

def preprocessing(img_path1,img_path2):

  input=[]
  labels=[]

  for filera in os.listdir(img_path1): # 이미지 이름 불러오기
    readpath = os.path.join(img_path1,filera) # 이미지 경로 불러오기
    img = cv2.imread(readpath) # 경로에서 이미지 읽어오기
    img = np.asarray(img) # 배열로 선언
    shapes = img.shape # shape 다 다름
    for i in range(0, shapes[0]-req_size+1, stride): # 0부터 shapes[0]-req_size+1까지의 거리를 stride 거리마다 i로 배정
      for j in range(0, shapes[1]-req_size+1, stride): # 0부터 shapes[1]-req_size+1까지의 거리를 stride 거리마다 i로 배정
        subimage_HR = img[i:i+req_size, j:j+req_size] # 고해상도 패치 추출
        #cv2_imshow(subimage_HR)
        subimage_LR = cv2.resize(subimage_HR,(inp_size,inp_size)) # 고해상도 패치를 resize 통해 저해상도 패치로 만듦
        input.append(subimage_LR) # 저해상도 패치를 input으로
        labels.append(subimage_HR) # 고해상도 패치를 label로

  counta=0
  for filera in os.listdir(img_path2): # 
    counta=counta+1
    if(counta==300): # 300번때 정지
      break
    readpath = os.path.join(img_path2,filera) 
    img = cv2.imread(readpath)
    img = np.asarray(img)
    shapes = img.shape
    for i in range(0, shapes[0]-req_size+1, stride):
      for j in range(0, shapes[1]-req_size+1,stride):
        subimage_HR = img[i:i+req_size, j:j+req_size]
        #cv2_imshow(subimage_HR)
        subimage_LR = cv2.resize(subimage_HR,(inp_size,inp_size))
        input.append(subimage_LR)
        labels.append(subimage_HR)

  return input, labels

def extract_patches(img, req_size, stride):
    patches = []
    shapes = img.shape
    for i in range(0, shapes[0] - req_size + 1, stride):
        for j in range(0, shapes[1] - req_size + 1, stride):
            patch = img[i:i + req_size, j:j + req_size]
            patches.append(patch)
    return patches

def preprocessing2(img_path1, img_path2, req_size, inp_size, stride):
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