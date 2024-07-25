import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
from tqdm import tqdm
import os
import random
from keras import backend as K

import tensorflow as tf
from keras.preprocessing.image import img_to_array


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def data_load(dir_path_in, ratio = None, shuffle = None):
    
    data_train = []
    data_val = []
    
    dir_list = os.listdir(dir_path_in) # path 안에 이미지들
    
    if shuffle is True :
        random.shuffle(dir_list)
    
    def data_spilit(dir_list,ratio):
        n = int(len(dir_list) * (1-ratio))
        dir_list_train,dir_list_val = dir_list[:n],dir_list[n:]
        return dir_list_train,dir_list_val

    dir_list_train,dir_list_val = data_spilit(dir_list,ratio=ratio)

    for i in range(len(dir_list_train)):
        img = np.asarray(Image.open(os.path.join(dir_path_in + dir_list[i])))
        data_train.append(img)

    data_train = np.array(data_train)

    for i in range(len(dir_list_val)):
        img = np.asarray(Image.open(os.path.join(dir_path_in + dir_list[i])))
        data_val.append(img)

    data_val = np.array(data_val)

    return data_train,data_val

def PSNR(y_true, y_pred):
  rmse = K.mean(K.square((y_pred-y_true)**2))
  psnr = 20*tf.experimental.numpy.log10(255./rmse)
  return psnr

def image_preprocessing(img_paths, upscale_factor, input_size, output_size, stride, pad):
    sub_lr_imgs = []
    sub_hr_imgs = []

    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        
        h = img.shape[0] - np.mod(img.shape[0], upscale_factor) # 176 - (176 % 3) => 175
        w = img.shape[1] - np.mod(img.shape[1], upscale_factor) # 197 - (197 % 3) ->195
        img = img[:h, :w, :] # (175, 195, 3)
        
        label = img.astype('float') / 255 # 고해상도
        temp_input = cv2.resize(label, dsize=(0,0), fx = 1/upscale_factor, fy = 1/upscale_factor,
                                interpolation = cv2.INTER_AREA) # 영역 보간법(이미지 축소)
        input = cv2.resize(temp_input, dsize=(0,0), fx = upscale_factor, fy = upscale_factor,
                            interpolation = cv2.INTER_CUBIC) # bicubic(이미지 확대)
        
        for h in range(0, input.shape[0] - input_size + 1, stride): # 저해상도 (0, 143 ,14)
            for w in range(0, input.shape[1] - input_size + 1, stride): # (0, 163, 14)
                sub_lr_img = input[h:h+input_size, w:w+input_size, :] # input[14:47, 14:47, :],...  => size 33
                sub_hr_img = label[h+pad:h+pad+output_size, w+pad:w+pad+output_size, :] # label[21:40, 21:40, :],... => size 19

                sub_lr_imgs.append(sub_lr_img)
                sub_hr_imgs.append(sub_hr_img)

    sub_lr_imgs = np.asarray(sub_lr_imgs)
    sub_hr_imgs = np.asarray(sub_hr_imgs)

    return sub_lr_imgs, sub_hr_imgs
